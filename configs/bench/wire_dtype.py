"""Production on four GPUs, real data, fp16 policy: gradient wire in fp16 and in fp32."""

from pathlib import Path

OUT = Path("runs/bench/wire_dtype.jsonl")
STEPS = 300
WARM_STEPS = 0
CASES = [
    {
        "recipe": "recipes_kvq_f16c256_production",
        "gpus": [0, 1, 2, 3],
        "data": "real",
        "knobs": {"REDUCE_DTYPE": wire},
    }
    for wire in ("fp16", None)
]
