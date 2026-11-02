"""The production recipe with the current sensecraft, one GPU, synthetic data."""

from pathlib import Path

OUT = Path("runs/bench/sensecraft_new.jsonl")
STEPS = 60
WARM_STEPS = 0
CASES = [{"recipe": "recipes_kvq_f16c256_production", "gpus": [0], "data": "synthetic"}]
