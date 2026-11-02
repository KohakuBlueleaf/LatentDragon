"""The four remaining recipes on the final setup, four GPUs, real data."""

from pathlib import Path

OUT = Path("runs/bench/final_rest.jsonl")
STEPS = 60
WARM_STEPS = 0
_FP16 = {"PRECISION": "fp16", "REDUCE_DTYPE": "bf16"}
CASES = [
    {"recipe": recipe, "gpus": [0, 1, 2, 3], "data": "real", "knobs": knobs}
    for recipe, knobs in (
        ("recipes_sdvae_f8c4", _FP16),
        ("bench_taming_active", _FP16),
        ("bench_production_active", _FP16),
        ("bench_r3gan_active", _FP16),
    )
]
