"""Reducer and bucket size on four GPUs, synthetic data, compiled, so only comm differs."""

from pathlib import Path

OUT = Path("runs/bench/reducer.jsonl")
STEPS = 150
_BASE = {"COMPILE": "model", "COMPILE_LOSS": True}
_BUCKETED = {"name": "bucketed", "bucket_mib": 8.0}
_ARMS = [
    {"REDUCER": "flat"},
    {"REDUCER": {**_BUCKETED, "bucket_mib": 2.0}},
    {"REDUCER": _BUCKETED},
    {"REDUCER": {**_BUCKETED, "bucket_mib": 25.0}},
    {"REDUCER": {**_BUCKETED, "bucket_mib": 64.0}},
    {"REDUCER": _BUCKETED, "REDUCE_DTYPE": "bf16"},
    {"REDUCER": _BUCKETED, "REDUCE_DTYPE": "fp16"},
]
CASES = [
    {
        "recipe": "recipes_kvq_f16c256_production",
        "gpus": [0, 1, 2, 3],
        "data": "synthetic",
        "knobs": {**_BASE, **arm},
    }
    for arm in _ARMS
]
