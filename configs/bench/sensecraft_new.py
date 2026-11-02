"""Production and kvq with the new sensecraft, one GPU each in parallel, synthetic data."""

from pathlib import Path

OUT = Path("runs/bench/sensecraft_new.jsonl")
STEPS = 60
WARM_STEPS = 0
CASES = [
    {"recipe": "recipes_kvq_f16c256", "gpus": [2], "data": "synthetic"},
]
