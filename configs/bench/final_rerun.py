"""Shipped recipes that shared a GPU with another job, on idle GPUs."""

from pathlib import Path

OUT = Path("runs/bench/final_rerun.jsonl")
STEPS = 200
CASES = [
    {"recipe": recipe, "gpus": [0, 1, 2, 3], "data": "real"}
    for recipe in ("recipes_kvq_f16c256_production", "recipes_kvq_f16c256")
]
