"""Question: which lines of a training step wait for the GPU?

Builds a recipe on one GPU, runs warm-up steps, then one step under sync debug `warn` and
prints each host sync with its Python stack. Also counts them. Real compute.

    kogine run experiments/throughput/sync_census.py --set RECIPE=bench_taming_active
"""

import json
import traceback
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
from kohakuengine import load_config_file

from latentdragon.training.build.assemble import build_bundles, build_ema, build_task
from latentdragon.training.drivers.native import Driver
from latentdragon.training.parallel import sync
from latentdragon.training.precision import PRECISION
from latentdragon.training.state import TrainState
from latentdragon.training.timing import StepTimer
from latentdragon_common.knobs import supplied_knobs
from latentdragon_common.registry import build

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "train" / "ae.py"
RECIPE: str = "bench_taming_active"
KNOBS: dict = {}
DEVICE: str = "cuda:0"
WARM: int = 8
OUT: Path = Path("runs/bench/sync_census.jsonl")


def knobs_of() -> dict:
    """Script defaults under the recipe, under `KNOBS`."""
    values = supplied_knobs(dict(load_config_file(SCRIPT).globals_dict))
    for folder in ("configs/ae/recipes", "configs/ae/bench"):
        path = ROOT / folder / f"{RECIPE}.py"
        if path.exists():
            values.update(supplied_knobs(dict(load_config_file(path).globals_dict)))
    values.pop("FRAMES", None)
    return {**values, "EPOCHS": 0, "MAX_STEPS": 100, **KNOBS}


def sync_sites(run: Callable[[], object]) -> dict[str, int]:
    """Host syncs raised by `run`, counted per innermost repo line of the stack."""
    sites: dict[str, int] = {}

    def note(message, *_) -> None:
        if "synchronizing" not in str(message):
            return
        frames = [f for f in traceback.extract_stack() if "latentdragon" in f.filename]
        where = "autograd thread"
        if frames:
            where = f"{Path(frames[-1].filename).name}:{frames[-1].lineno} {frames[-1].line}"
        sites[where] = sites.get(where, 0) + 1

    shown = warnings.showwarning
    warnings.showwarning = note
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("always")
            warnings.showwarning = note
            run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
        warnings.showwarning = shown
    return sites


def main() -> None:
    """Print every synchronization of one step, and how many there were."""
    knobs = knobs_of()
    torch.cuda.set_device(DEVICE)
    task = build_task(knobs, DEVICE)
    bundles = build_bundles(task, knobs, 1000)
    precision = build(knobs["PRECISION"], PRECISION)
    ema = build_ema(task, knobs)
    timer = StepTimer(0, DEVICE)
    driver = Driver(
        task, bundles, ema, precision, DEVICE, knobs["GRAD_ACC"], sync, timer
    )
    task.train()

    size = knobs["SIZE"]
    batch = torch.rand(knobs["BATCH_SIZE"], 3, size, size, device=DEVICE) * 2 - 1
    state = TrainState()
    for _ in range(WARM):
        driver.step(batch, state)
        state.step += 1

    # Metric steps run extra code, so census both kinds
    rows = []
    for step in (state.step + 1, state.step + 2):
        state.step = step * 10 if knobs["METRICS_EVERY"] == 1 else step
        sites = sync_sites(lambda: driver.step(batch, state))
        rows.append({"recipe": RECIPE, "step": state.step, "sites": sites})
        print(f"step {state.step}: {sum(sites.values())} syncs", flush=True)
        for site, count in sorted(sites.items()):
            print(f"  {count:3d} {site}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
