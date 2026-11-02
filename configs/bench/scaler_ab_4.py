"""Production on four GPUs, real data: scaler A/B."""

from pathlib import Path

OUT = Path("runs/bench/scaler_ab_4.jsonl")
STEPS = 300
WARM_STEPS = 0
CASES = [
    {"recipe": "recipes_kvq_f16c256_production", "gpus": [0, 1, 2, 3], "data": "real"}
]
