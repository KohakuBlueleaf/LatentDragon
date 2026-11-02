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
CENSUS_STEPS: int = 4  # consecutive steps counted after the warm-up
RAISE: bool = False  # raise at the first sync and show the forward stack of its op
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


OWN_CODE = ("latentdragon", "sensecraft")


def raise_at_sync(run: Callable[[], object]) -> None:
    """Run with every host sync raising, and the forward stack of a failing backward node shown."""
    torch.cuda.set_sync_debug_mode("error")
    try:
        with torch.autograd.detect_anomaly(check_nan=False):
            run()
    except RuntimeError as err:
        frames = traceback.extract_tb(err.__traceback__)
        own = [f for f in frames if any(name in f.filename for name in OWN_CODE)]
        for frame in own[-4:] or frames[-4:]:
            print(
                f"  {Path(frame.filename).name}:{frame.lineno} {frame.line}", flush=True
            )
        print(f"sync raised: {str(err)[:400]}", flush=True)
    finally:
        torch.cuda.set_sync_debug_mode("default")


def sync_sites(run: Callable[[], object]) -> dict[str, int]:
    """Host syncs raised by `run`, counted per innermost repo line of the stack."""
    sites: dict[str, int] = {}

    def note(message, *_) -> None:
        # PyTorch's notice that the debug mode is a prototype is not a sync
        text = str(message)
        if "synchronizing" not in text or "prototype" in text:
            return
        # The innermost frame of our code or sensecraft; none means the C++ autograd engine
        stack = traceback.extract_stack()[:-1]
        frames = [f for f in stack if any(name in f.filename for name in OWN_CODE)]
        where = "autograd thread"
        if frames:
            frame = frames[-1]
            where = f"{Path(frame.filename).name}:{frame.lineno} {frame.line}"
        elif stack:
            frame = stack[-1]
            where = f"no own frame, innermost {Path(frame.filename).name}:{frame.lineno} {frame.line}"
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
    for step in range(state.step + 1, state.step + 1 + CENSUS_STEPS):
        state.step = step * 10 if knobs["METRICS_EVERY"] == 1 else step
        if RAISE:
            raise_at_sync(lambda: driver.step(batch, state))
            continue
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
