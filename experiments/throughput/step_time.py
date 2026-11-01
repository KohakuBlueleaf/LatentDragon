"""Question: how fast is one training step, and where does the time go?

Runs a recipe's architecture on noise for STEPS steps per arm, draining the device at
phase edges, and writes steps per second plus the phase breakdown. Arms flip GRAD_CKPT,
COMPILE and KERNELS. Real compute: run only when approved.

    kogine run experiments/throughput/step_time.py --set DEVICE=cuda:0
"""

import json
import time
from pathlib import Path

import torch
from sensecraft.loss import GeneralConfig, LPIPSConfig

from latentdragon.training.build.assemble import build_bundles, build_task
from latentdragon.training.drivers.native import Driver
from latentdragon.training.ema import NoEMA
from latentdragon.training.parallel import sync
from latentdragon.training.precision import PRECISION as PRECISIONS
from latentdragon.training.state import TrainState
from latentdragon.training.timing import StepTimer
from latentdragon_common.registry import build

DEVICE = "cuda:0"
PRESET = "kvq_f16c256"
OUT = "runs/experiments/step_time.jsonl"
BATCH = 32
SIZE = 256
STEPS = 50
WARMUP = 10
ARMS = [
    {"GRAD_CKPT": True, "COMPILE": False, "KERNELS": "auto"},
    {"GRAD_CKPT": False, "COMPILE": False, "KERNELS": "auto"},
    {"GRAD_CKPT": False, "COMPILE": True, "KERNELS": "auto"},
    {"GRAD_CKPT": False, "COMPILE": False, "KERNELS": "eager"},
]
BASE = {
    "ARCH": PRESET,
    "ARCH_OVERRIDES": {},
    "INIT_FROM": None,
    "GAN_SCHEME": {"name": "taming", "disc_start": 0},
    "CRITIC": {"name": "modern_patchgan", "ndf": 64, "n_layers": 3, "norm": "rms"},
    "OBJECTIVE": None,
    "LOSS_CONFIGS": [
        GeneralConfig("mse", weight=1.0),
        LPIPSConfig(net="vgg", weight=0.5),
    ],
    "LATENT_VIEWS": [],
    "LATENT_TERMS": [],
    "TEACHERS": {},
    "AUX_WEIGHTS": {"vq": 1.0, "kl": 0.0},
    "FREEZE": [],
    "ENC_LR_SCALE": 1.0,
    "VQ_LR_SCALE": 1.0,
    "DISC_LR_SCALE": 1.0,
    "METRICS_EVERY": 10,
    "USAGE_WINDOW": 1000,
    "KERNEL_TUNE": "on",
    "PRECISION": "bf16",
    "REDUCE_DTYPE": None,
    "SCHEDULER_CONFIG": {"mode": "constant", "value": 1.0, "end": -1},
    "OPTIMIZER": "adam",
    "LR": 1e-4,
    "BETAS": (0.5, 0.95),
    "WEIGHT_DECAY": 0.0,
    "GRAD_CLIP": 1.0,
}


def run_arm(arm: dict) -> dict:
    """Steps per second, peak memory and phase breakdown of one arm."""
    # Task, driver and a fixed noise batch
    knobs = {**BASE, **arm}
    task = build_task(knobs, DEVICE)
    bundles = build_bundles(task, knobs, STEPS + WARMUP)
    timer = StepTimer(1, DEVICE)
    precision = build(knobs["PRECISION"], PRECISIONS)
    driver = Driver(task, bundles, NoEMA(), precision, DEVICE, 1, sync, timer)
    task.train()
    x = (torch.rand(BATCH, 3, SIZE, SIZE, device=DEVICE) * 2 - 1).detach()
    state = TrainState()

    # Warm-up steps are not timed; drop their phase records
    for _ in range(WARMUP):
        driver.step(x, state)
        state.step += 1
    timer.seconds.clear()
    timer.counts.clear()

    # Timed steps
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(STEPS):
        timer.start_step(state.step)
        driver.step(x, state)
        state.step += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - begin
    peak = torch.cuda.max_memory_allocated(DEVICE) / 2**30

    return {
        **arm,
        "steps_per_second": STEPS / elapsed,
        "peak_gib": peak,
        "phases": timer.report(),
    }


def main():
    """Append one result row per arm to `OUT`."""
    path = Path(OUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for arm in ARMS:
            handle.write(json.dumps(run_arm(arm)) + "\n")


if __name__ == "__main__":
    main()
