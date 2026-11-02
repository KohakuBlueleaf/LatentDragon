"""The fastest setup on four GPUs with real data."""

from pathlib import Path

OUT = Path("runs/bench/final_multi.jsonl")
STEPS = 200
_FAST = {"COMPILE": "model", "COMPILE_LOSS": True}
_RECIPES = [
    ("recipes_kvq_f16c256_production", False),
    ("recipes_kvq_f16c256", False),
    ("recipes_latentmaid_f8c4", True),
    ("recipes_latentmaid_0222_f8c4", False),
    ("recipes_sdvae_f8c4", True),
    ("bench_taming_active", False),
    ("bench_production_active", False),
    ("bench_r3gan_active", True),
]
CASES = [
    {
        "recipe": recipe,
        "gpus": [0, 1, 2, 3],
        "data": "imagenet",
        "knobs": {**_FAST, "GRAD_CKPT": ckpt},
    }
    for recipe, ckpt in _RECIPES
]
