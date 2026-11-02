"""Question: what does a recipe actually train at, and where does its step go?

Runs recipes through the real entry point with validation, samples and checkpoints off, and
appends one JSON row per case: step interval, phases, data wait, peak memory and GPU busy
share. The cases live in a sweep config: `CASES`, each naming a recipe by its path under
`configs/` without `.py`. Real compute.

    kogine run experiments/throughput/recipes.py --config <sweep config>
"""

import json
import os
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

# Read at import by huggingface_hub, and inherited by spawned ranks
STAND_IN_HUB = Path(
    "runs/bench_hub"
).resolve()  # a random DINOv3 ViT-B, for timing only
os.environ["HF_HOME"] = str(STAND_IN_HUB)
os.environ["HF_HUB_OFFLINE"] = "1"
# Converted stand-in weights stay beside the stand-in hub, never in the user cache
os.environ["SENSECRAFT_CACHE"] = str(STAND_IN_HUB / "sensecraft")

import torch  # noqa: E402
from kohakuengine import load_config_file  # noqa: E402
from transformers import DINOv3ViTConfig, DINOv3ViTModel  # noqa: E402

from latentdragon.training.entry import run  # noqa: E402
from latentdragon_common.knobs import supplied_knobs  # noqa: E402

DINO_REPO = "facebook/dinov3-vitb16-pretrain-lvd1689m"

TREE: Path = Path(
    "."
)  # checkout holding the recipes and script; PYTHONPATH picks its `src`
OUT: Path = Path("runs/bench/recipes.jsonl")
REAL_SOURCES: list | None = (
    None  # replaces TRAIN_SOURCES on "real" cases; None keeps the recipe's
)
STEPS: int = 120
WARM_STEPS: int = 6
SCRATCH: Path = Path("runs/bench_tmp")
CASES: list[dict] = []
SHARD: int | None = None  # run every 4th case on this GPU, for one-GPU sweeps


def stand_in_hub() -> None:
    """Write a random DINOv3 ViT-B into the private Hugging Face cache.

    The real repo is gated. Speed does not depend on the weight values, and the private
    folder keeps random weights out of the machine's real cache.
    """
    snapshot = STAND_IN_HUB / "hub" / f"models--{DINO_REPO.replace('/', '--')}"
    files = snapshot / "snapshots" / "standin"
    if not files.exists():
        config = DINOv3ViTConfig(
            image_size=224,
            patch_size=16,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            num_register_tokens=4,
        )
        files.mkdir(parents=True)
        DINOv3ViTModel(config).save_pretrained(files)
        (snapshot / "refs").mkdir()
        (snapshot / "refs" / "main").write_text("standin")


def script_path() -> Path:
    """The train script of `TREE`."""
    return TREE.resolve() / "scripts" / "train" / "ae.py"


def recipe_path(name: str) -> Path:
    """The config file of a recipe or bench config: `name` is a path under configs/, no .py."""
    path = script_path().parents[2] / "configs" / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def script_defaults() -> dict:
    """The train script's own knobs."""
    return supplied_knobs(dict(load_config_file(script_path()).globals_dict))


def knobs_of(case: dict, folder: str, steps: int) -> dict:
    """The knobs of one bench run."""
    values = script_defaults()
    recipe = load_config_file(recipe_path(case["recipe"]))
    values.update(supplied_knobs(dict(recipe.globals_dict)))
    frames = values.pop("FRAMES", 0)
    values.update(
        NAME="bench",
        RUN_ID="run",
        RUNS_DIR=folder,
        DEVICES=case["gpus"],
        MAX_STEPS=steps,
        EPOCHS=0,
        VAL_EVERY=0,
        SAMPLE_EVERY=0,
        CKPT_EVERY=10**9,
        WANDB_PROJECT=None,
        FLUSH_EVERY=20,
    )
    if case["data"] == "synthetic":
        size = values["SIZE"]
        shape = (frames, 3, size, size) if frames else (3, size, size)
        values["TRAIN_SOURCES"] = [{"name": "synthetic", "shape": shape}]
        values["NUM_WORKERS"] = 0
    elif REAL_SOURCES is not None:
        values["TRAIN_SOURCES"] = REAL_SOURCES
    values.update(case.get("knobs", {}))
    return values


class BusySampler:
    """Mean GPU utilization and memory of `gpus`, read from `nvidia-smi` once a second."""

    def __init__(self, gpus: list[int]):
        self.gpus = gpus
        self.samples: list[tuple[float, float]] = []
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        query = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used",
            "--format=csv,noheader,nounits",
            "-i",
            ",".join(str(g) for g in self.gpus),
        ]
        while self.running:
            lines = subprocess.run(query, capture_output=True, text=True).stdout
            rows = [line.split(", ") for line in lines.strip().splitlines()]
            if rows:
                busy = sum(float(r[0]) for r in rows) / len(rows)
                memory = max(float(r[1]) for r in rows)
                self.samples.append((busy, memory))
            time.sleep(1.0)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.running = False
        self.thread.join()

    def summary(self) -> dict:
        """Busy share over the second half of the run, and the peak memory used."""
        kept = self.samples[len(self.samples) // 2 :] or [(0.0, 0.0)]
        peak = max((m for _, m in self.samples), default=0.0)
        return {
            "gpu_busy": sum(b for b, _ in kept) / len(kept),
            "smi_peak_gib": peak / 1024,
        }


def run_once(case: dict, steps: int) -> tuple[dict, dict, float, dict]:
    """One run of `case`; returns its knobs, timing, wall seconds and busy summary."""
    SCRATCH.mkdir(parents=True, exist_ok=True)
    folder = tempfile.mkdtemp(dir=SCRATCH)
    values = knobs_of(case, folder, steps)
    wall = time.perf_counter()
    with BusySampler(case["gpus"]) as busy:
        run(values, str(script_path()))
    wall = time.perf_counter() - wall
    timing = json.loads(Path(folder, "bench", "run", "timing.json").read_text())
    shutil.rmtree(folder)
    return values, timing, wall, busy.summary()


def run_case(case: dict) -> dict:
    """A short warm run for kernel tuning and caches, then the timed run."""
    if WARM_STEPS:
        run_once(case, WARM_STEPS)
    values, timing, wall, busy = run_once(case, STEPS)
    images = values["BATCH_SIZE"] * len(case["gpus"])
    return {
        **case,
        "wall_seconds": wall,
        "images_per_second": images / timing["step_median"],
        "per_gpu_batch": values["BATCH_SIZE"],
        "phases": timing,
        "step_median": timing["step_median"],
        **busy,
    }


def shard_of(cases: list[dict]) -> list[dict]:
    """Every case, or with `SHARD` set this GPU's quarter of them."""
    if SHARD is None:
        return cases
    return [{**case, "gpus": [SHARD]} for case in cases[SHARD::4]]


def main() -> None:
    """Append one row per case to `OUT`."""
    stand_in_hub()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for case in shard_of(CASES):
        try:
            row = run_case(case)
        except (RuntimeError, ValueError, SystemExit) as err:
            row = {**case, "error": f"{type(err).__name__}: {err}"[:300]}
            with OUT.open("a", encoding="utf-8") as file:
                file.write(json.dumps(row) + "\n")
            print(f"{case}: {row['error']}", flush=True)
            torch.cuda.empty_cache()
            continue
        with OUT.open("a", encoding="utf-8") as file:
            file.write(json.dumps(row) + "\n")
        print(f"{case}: {row['images_per_second']:.1f} img/s", flush=True)


if __name__ == "__main__":
    main()
