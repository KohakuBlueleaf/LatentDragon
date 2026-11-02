"""The active taming recipes under compile, after the retained backward fix."""

from pathlib import Path

OUT = Path("runs/bench/taming_compiled.jsonl")
STEPS = 100
_FAST = {"COMPILE": "model", "COMPILE_LOSS": True}
CASES = [
    {
        "recipe": recipe,
        "gpus": [0],
        "data": "synthetic",
        "knobs": {**_FAST, "GRAD_CKPT": ckpt},
    }
    for recipe in ("bench_taming_active", "bench_production_active")
    for ckpt in (True, False)
]
