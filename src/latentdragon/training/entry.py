"""`run(globals(), __file__)`: validate, spawn, assemble and drive one run."""

import functools
import os
import socket
import sys
import warnings
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from latentdragon.domain import DOMAIN
from latentdragon.eval import ReconEvaluator
from latentdragon.model import build_autoencoder
from latentdragon.model.core import tree
from latentdragon.training.build.assemble import (
    build_bundles,
    build_ema,
    build_task,
    total_steps,
)
from latentdragon.training.callbacks.runner import CallbackRunner
from latentdragon.training.callbacks.validation import (
    CheckpointCallback,
    SampleCallback,
    ValidationCallback,
)
from latentdragon.training.checkpoint import CheckpointManager, latest
from latentdragon.training.checks import all_problems, compile_notes
from latentdragon.training.drivers.native import Driver
from latentdragon.training.logging.metrics import MetricLog
from latentdragon.training.logging.sinks import (
    ConsoleSink,
    JsonlSink,
    ProgressBar,
    WandbSink,
)
from latentdragon.training.logging.summary import summarize
from latentdragon.training.parallel import sync
from latentdragon.training.precision import PRECISION
from latentdragon.training.run_dir import (
    drift_problems,
    jsonable,
    new_run_id,
    open_run,
    resolve_run,
    write_json_atomic,
)
from latentdragon.training.state import TrainState
from latentdragon.training.timing import StepTimer
from latentdragon_common.knobs import (
    raise_problems,
    script_knobs,
    supplied_knobs,
    unknown_notes,
)
from latentdragon_common.log import get_logger
from latentdragon_common.registry import build
from latentdragon_data import SOURCE, Concat, epoch_batches

log = get_logger(__name__)

# Triton's current default, made explicit: its next release turns FP fusion off by default
FP_FUSION = ("TRITON_DEFAULT_FP_FUSION", "1")
# An idle critic's schedule keeps counting global steps before its optimizer first steps
SCHEDULE_FIRST = r"Detected call of `lr_scheduler.step\(\)` before `optimizer.step\(\)`"


def free_port() -> int:
    """An unused local TCP port for the process group."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def check(knobs: dict, script: str) -> None:
    """Report every config problem at once, before any device work."""
    for note in unknown_notes(set(knobs), script_knobs(script)):
        log.warning("config", note=note)
    problems = []
    try:
        with torch.device("meta"):
            model = build_autoencoder(knobs["ARCH"])
    except (KeyError, TypeError, ValueError, AttributeError, ImportError) as err:
        raise_problems([*problems, f"ARCH: {err}"])
    problems += all_problems(knobs, model)
    problems += freeze_problems(model, knobs["FREEZE"])
    if knobs["RESUME"] is not None:
        problems += drift_problems(resolve_run(knobs), knobs)
    raise_problems(problems)
    for note in compile_notes(knobs, model):
        log.warning("compile", note=note)


def freeze_problems(model, paths: list[str]) -> list[str]:
    """`FREEZE` paths that name no submodule of the model."""
    problems = []
    for path in paths:
        try:
            model.get_submodule(path)
        except AttributeError:
            problems.append(f"FREEZE path {path!r} is not a model part")
    return problems


def run(values: dict, script: str) -> None:
    """Script entry point: validate, then train on every device in `DEVICES`."""
    knobs = supplied_knobs(values)
    os.environ.setdefault(*FP_FUSION)
    check(knobs, script)
    knobs["RUN_ID"] = knobs["RUN_ID"] or new_run_id()
    devices = knobs["DEVICES"]

    # Several devices: one spawned process per device on a local rendezvous port.
    if len(devices) > 1:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ["MASTER_PORT"] = str(free_port())
        mp.spawn(worker, args=(len(devices), knobs, script), nprocs=len(devices))
        return
    worker(0, 1, knobs, script)


def _device(knobs: dict, rank: int) -> str:
    """The device string of `rank`; a bare index means that CUDA device."""
    device = knobs["DEVICES"][rank]
    return device if isinstance(device, str) else f"cuda:{device}"


def _sources(specs: list, domain, size: int, train: bool):
    """One data source per spec, concatenated when there are several."""
    built = [
        build(spec, SOURCE, transform=domain.transforms(size, train)) for spec in specs
    ]
    return built[0] if len(built) == 1 else Concat(built)


def _evaluator(k, domain, device, precision, rank, world):
    """The reconstruction evaluator over this rank's strided share of `VAL_SOURCE`."""
    source = _sources([k["VAL_SOURCE"]], domain, k["SIZE"], False)
    indices = list(range(rank, len(source), world))
    return ReconEvaluator(
        source,
        k["VAL_BATCH_SIZE"],
        k["VAL_WORKERS"],
        device,
        precision,
        sampler=indices,
    )


def _sinks(k: dict, run_dir, rank: int, script: str) -> list:
    """Metric sinks; only rank 0 writes, and wandb only when a project is set.

    An interactive terminal gets a progress bar; a log file gets one line per flush.
    """
    if rank != 0:
        return []
    sinks = [JsonlSink(run_dir / "metrics.jsonl")]
    sinks.append(ConsoleSink() if not sys.stderr.isatty() else None)
    if k["WANDB_PROJECT"] is not None:
        files = [Path(script), run_dir / "config.json", run_dir / "git.json"]
        sinks.append(
            WandbSink(
                k["WANDB_PROJECT"],
                k["NAME"],
                jsonable(k),
                run_dir,
                [path for path in files if path.exists()],
                run_id=k["WANDB_RUN_ID"],
                resume=k["WANDB_RESUME"] or ("allow" if k["WANDB_RUN_ID"] else None),
                offline=k["WANDB_OFFLINE"],
            )
        )
    return [sink for sink in sinks if sink is not None]


def _summary(k, task, run_dir, sinks) -> None:
    """Print the module summary, keep it in the run folder and hand totals to wandb."""
    table, totals = summarize(task.summary_modules(), k["SUMMARY_DEPTH"])
    print(table, flush=True)
    path = run_dir / "model_summary.txt"
    path.write_text(table + "\n", encoding="utf-8")
    for sink in sinks:
        if isinstance(sink, WandbSink):
            sink.summary(totals, path)


def _restore(k, run_dir, manager, state, task, bundles, ema, rank) -> None:
    """Continue from the newest checkpoint of a resumed run."""
    if k["RESUME"] is None:
        return
    found = latest(run_dir / "checkpoints")
    if found is None:
        return
    manager.restore(found, state, task.saved_modules(), bundles, ema)
    if rank == 0:
        JsonlSink(run_dir / "metrics.jsonl").truncate_after(state.step)


def _callbacks(k, task, bundles, ema, manager, metric_log, env) -> list:
    """Checkpoint, validation and sample callbacks selected by their cadence knobs."""
    device, rank, world, domain, precision, run_dir, source = env
    callbacks = [CheckpointCallback(k["CKPT_EVERY"], manager, task, bundles, ema)]

    if k["VAL_SOURCE"] is not None and k["VAL_EVERY"] > 0:
        evaluator = _evaluator(k, domain, device, precision, rank, world)
        callbacks.append(
            ValidationCallback(k["VAL_EVERY"], evaluator, task, ema, sync, metric_log)
        )

    if k["SAMPLE_EVERY"] > 0 and rank == 0:
        # Sample grids: the step's own batch, and the first (up to) 8 training items held fixed.
        held = tree.stack([source[i] for i in range(min(8, len(source)))])
        device_type = torch.device(device).type
        callbacks.append(
            SampleCallback(
                k["SAMPLE_EVERY"],
                task,
                task.to_device(held, device),
                run_dir / "samples",
                functools.partial(precision.autocast, device_type),
                metric_log.image,
            )
        )
    return callbacks


def _start(rank: int, world: int, k: dict) -> tuple:
    """Process group, device, seed and run folder of this rank."""
    device = _device(k, rank)
    os.environ["RANK"] = str(rank)
    warnings.filterwarnings("ignore", message=SCHEDULE_FIRST)
    if "cuda" in device:
        torch.cuda.set_device(device)
    if world > 1:
        backend = "nccl" if "cuda" in device else "gloo"
        timeout = timedelta(hours=k["DIST_TIMEOUT_HOURS"])
        # Bind NCCL to this rank's device; it otherwise guesses cuda:<rank>, wrong for DEVICES=[2, 3]
        bound = torch.device(device) if backend == "nccl" else None
        dist.init_process_group(
            backend, rank=rank, world_size=world, timeout=timeout, device_id=bound
        )
    torch.set_num_threads(k["CPU_THREADS"])
    torch.backends.cudnn.benchmark = k["CUDNN_BENCHMARK"]
    torch.manual_seed(k["SEED"])
    run_dir = resolve_run(k)
    if rank == 0:
        open_run(run_dir, k)
    sync.barrier()
    return device, run_dir


def worker(rank: int, world: int, k: dict, script: str = "") -> None:
    """One rank of a run: assemble once, then hand over to the driver."""
    device, run_dir = _start(rank, world, k)

    # Data, task, optimizers and EMA.
    domain = build(k["ARCH"].get("domain", "image"), DOMAIN)
    precision = build(k["PRECISION"], PRECISION)
    source = _sources(k["TRAIN_SOURCES"], domain, k["SIZE"], True)
    steps = total_steps(k, len(source), world)
    task = build_task(k, device)
    torch.manual_seed(k["SEED"] + rank)
    bundles = build_bundles(task, k, steps)
    ema = build_ema(task, k)
    task.train()

    # Checkpointing, resume, logging and callbacks.
    state = TrainState()
    manager = CheckpointManager(run_dir / "checkpoints", k["CKPT_KEEP"], rank)
    _restore(k, run_dir, manager, state, task, bundles, ema, rank)
    metric_log = MetricLog(_sinks(k, run_dir, rank, script), k["FLUSH_EVERY"])
    if rank == 0:
        _summary(k, task, run_dir, metric_log.sinks)
    progress = None
    if rank == 0 and sys.stderr.isatty():
        progress = ProgressBar(steps, state.step, k["NAME"])
        metric_log.sinks.append(progress)
    if rank == 0 and k["WANDB_WATCH"] > 0:
        for sink in metric_log.sinks:
            if isinstance(sink, WandbSink):
                sink.watch(task.model, k["WANDB_WATCH"])
    env = (device, rank, world, domain, precision, run_dir, source)
    runner = CallbackRunner(_callbacks(k, task, bundles, ema, manager, metric_log, env))

    # Batches continue from the restored epoch and batch position.
    batches = epoch_batches(
        source,
        k["BATCH_SIZE"],
        k["NUM_WORKERS"],
        rank,
        world,
        k["SEED"],
        state.epoch,
        state.batches,
        pin_memory="cuda" in device,
    )

    timer = StepTimer(k["TIMING_EVERY"], device)
    driver = Driver(task, bundles, ema, precision, device, k["GRAD_ACC"], sync, timer)

    images_per_step = k["BATCH_SIZE"] * world

    def end_step(current: TrainState, metrics: dict, batch) -> None:
        """Log the step's smoothed metrics with epoch and host timing, then fire callbacks."""
        values = task.smooth_metrics(current.step, metrics)
        values["epoch"] = current.epoch
        if len(timer.starts) > 1:
            seconds = timer.starts[-1] - timer.starts[-2]
            values["perf/step_time"] = seconds
            values["perf/samples_per_sec"] = images_per_step / seconds
        metric_log.record(current.step, values)
        if progress is not None:
            progress.advance(current.step)
        runner.step_end(current, batch)

    # Train, then flush callbacks and logs and write the timing report.
    driver.run(batches, state, steps, end_step)
    runner.finish(state)
    metric_log.close()
    if rank == 0:
        write_json_atomic(run_dir / "timing.json", timer.report())
    log.info("finished", step=state.step, run=str(run_dir))
    if world > 1:
        dist.destroy_process_group()
