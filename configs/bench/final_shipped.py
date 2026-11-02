"""The shipped recipes as they stand, four GPUs, real data."""

from pathlib import Path

OUT = Path("runs/bench/final_shipped2.jsonl")
STEPS = 200
CASES = [
    {"recipe": recipe, "gpus": [0, 1, 2, 3], "data": "real"}
    for recipe in (
        "recipes_kvq_f16c256_production",
        "recipes_kvq_f16c256",
        "recipes_latentmaid_0222_f8c4",
        "recipes_latentmaid_f8c4",
        "recipes_sdvae_f8c4",
        "bench_taming_active",
        "bench_production_active",
        "bench_r3gan_active",
    )
]
