"""Four GPUs: what costs fp16 its speed, separating precision, wire dtype and data."""

from pathlib import Path

OUT = Path("runs/bench/fp16_multi.jsonl")
STEPS = 150
_RECIPE = "recipes_kvq_f16c256_production"
_ARMS = [
    ("synthetic", {"PRECISION": "bf16", "REDUCE_DTYPE": "bf16"}),
    ("synthetic", {"PRECISION": "bf16", "REDUCE_DTYPE": None}),
    ("synthetic", {"PRECISION": "fp16", "REDUCE_DTYPE": None}),
    ("real", {"PRECISION": "bf16", "REDUCE_DTYPE": "bf16"}),
    ("real", {"PRECISION": "bf16", "REDUCE_DTYPE": None}),
]
CASES = [
    {"recipe": _RECIPE, "gpus": [0, 1, 2, 3], "data": data, "knobs": knobs}
    for data, knobs in _ARMS
]
