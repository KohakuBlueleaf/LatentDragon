"""Question: which ops hold the GPU during a training step?

Profiles a few steps of a recipe with CUPTI and prints the ops by device time. Weights and
data are synthetic; kernels and compile follow `KNOBS`. Real compute.

    kogine run experiments/throughput/op_profile.py --set RECIPE=recipes_kvq_f16c256_production
"""

import json
from pathlib import Path

import torch
from kohakuengine import load_config_file
from torch.profiler import ProfilerActivity, profile

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
RECIPE: str = "recipes_kvq_f16c256_production"
KNOBS: dict = {}
DEVICE: str = "cuda:0"
WARM: int = 6
STEPS: int = 4
TOP: int = 30
OUT: Path = Path("runs/bench/op_profile.jsonl")


def knobs_of() -> dict:
    """Script defaults under the recipe, under `KNOBS`."""
    values = supplied_knobs(dict(load_config_file(SCRIPT).globals_dict))
    for folder in ("configs/ae/recipes", "configs/ae/bench"):
        path = ROOT / folder / f"{RECIPE}.py"
        if path.exists():
            values.update(supplied_knobs(dict(load_config_file(path).globals_dict)))
    values.pop("FRAMES", None)
    return {**values, "EPOCHS": 0, "MAX_STEPS": 100, **KNOBS}


def main() -> None:
    """Print and append the top ops by device time per step."""
    knobs = knobs_of()
    torch.cuda.set_device(DEVICE)
    task = build_task(knobs, DEVICE)
    bundles = build_bundles(task, knobs, 1000)
    precision = build(knobs["PRECISION"], PRECISION)
    timer = StepTimer(0, DEVICE)
    driver = Driver(
        task,
        bundles,
        build_ema(task, knobs),
        precision,
        DEVICE,
        knobs["GRAD_ACC"],
        sync,
        timer,
    )
    task.train()

    size = knobs["SIZE"]
    batch = torch.rand(knobs["BATCH_SIZE"], 3, size, size, device=DEVICE) * 2 - 1
    state = TrainState()
    for _ in range(WARM):
        driver.step(batch, state)
        state.step += 1
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        for _ in range(STEPS):
            driver.step(batch, state)
            state.step += 1
        torch.cuda.synchronize()

    rows = [
        {
            "op": event.key,
            "device_ms": event.device_time_total / 1e3 / STEPS,
            "self_device_ms": event.self_device_time_total / 1e3 / STEPS,
            "calls": event.count / STEPS,
        }
        for event in trace.key_averages()
    ]
    rows.sort(key=lambda row: -row["self_device_ms"])
    total = sum(row["self_device_ms"] for row in rows)
    print(f"{RECIPE}: {total:.1f} ms device time per step", flush=True)
    for row in rows[:TOP]:
        share = row["self_device_ms"] / total * 100
        print(
            f"{row['self_device_ms']:8.2f} ms {share:5.1f}% "
            f"{row['calls']:7.1f} calls  {row['op'][:70]}",
            flush=True,
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as file:
        file.write(
            json.dumps({"recipe": RECIPE, "knobs": KNOBS, "ops": rows[:80]}) + "\n"
        )


if __name__ == "__main__":
    main()
