"""Tiny run settings shared by the behavior checks."""

import importlib.util
import tempfile
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image
from sensecraft import Term

from latentdragon.training.entry import run
from latentdragon_common.knobs import supplied_knobs

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "train" / "ae.py"
TINY_ARCH = {
    "latent_dim": 4,
    "encoder": "conv",
    "encoder_blocks": ((2, 8, 1),),
    "encoder_attn": None,
    "trunk": "conv",
    "trunk_blocks": ((2, 8, 1),),
    "trunk_attn": None,
    "bottleneck": {"name": "vq", "quantizer": {"name": "kohaku", "n_embed": 8}},
}
TINY_CRITIC = {"name": "modern_patchgan", "ndf": 4, "n_layers": 1}


def write_images(root: Path, count: int = 8, size: int = 16) -> str:
    """Random images in a folder an `image_folder` source can read."""
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for index in range(count):
        data = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
        Image.fromarray(data).save(root / f"{index}.png")
    return str(root)


def knobs(folder: Path, **changes) -> dict:
    """The train script's own defaults with a two-step CPU run on top."""
    spec = importlib.util.spec_from_file_location("ae_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    values = supplied_knobs(vars(module))
    values.update(
        NAME="check",
        RUN_ID="run",
        RUNS_DIR=str(folder / "runs"),
        TRAIN_SOURCES=[{"name": "image_folder", "root": write_images(folder / "d")}],
        SIZE=16,
        NUM_WORKERS=0,
        ARCH=TINY_ARCH,
        LOSS_CONFIGS=[Term("mse")],
        BATCH_SIZE=2,
        MAX_STEPS=2,
        PRECISION="fp32",
        DEVICES=["cpu"],
        LOG_EVERY=1,
        METRICS_EVERY=1,
        SAMPLE_EVERY=0,
        VAL_EVERY=0,
        CKPT_EVERY=2,
        KERNELS="eager",
    )
    values.update(changes)
    return values


def run_dir_of(values: dict) -> Path:
    """The folder a run with these knobs writes to."""
    return Path(values["RUNS_DIR"]) / values["NAME"] / values["RUN_ID"]


def train(values: dict) -> Path:
    """Run the train script's entry point and return the run folder."""
    run(values, str(SCRIPT))
    return run_dir_of(values)


def run_checks(*checks: Callable[[Path], None]) -> None:
    """Call each check with its own temporary folder; the first failure stops the script."""
    for check in checks:
        with tempfile.TemporaryDirectory() as folder:
            check(Path(folder))
        print(f"ok  {check.__name__}")
