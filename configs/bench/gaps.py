"""One GPU each: reduce-overhead with the flat reducer, and 0222 on real and synthetic data."""

from pathlib import Path

OUT = Path("runs/bench/gaps.jsonl")
STEPS = 60
WARM_STEPS = 0
_FAST = {"COMPILE": "model", "COMPILE_LOSS": True}
CASES = [
    {
        "recipe": "recipes_kvq_f16c256_production",
        "gpus": [0],
        "data": "synthetic",
        "knobs": {**_FAST, "COMPILE_MODE": "reduce-overhead", "REDUCER": "flat"},
    },
    {"recipe": "recipes_latentmaid_0222_f8c4", "gpus": [1], "data": "real"},
    {"recipe": "recipes_latentmaid_0222_f8c4", "gpus": [2], "data": "synthetic"},
]
