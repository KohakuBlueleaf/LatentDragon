"""KVQ recipes on the gemm codebook kernel, compiled, GPU taken from the shard."""

from pathlib import Path

OUT = Path("runs/bench/kvq_gemm.jsonl")
STEPS = 100
_LOSS = {"COMPILE": "model", "COMPILE_LOSS": True}
CASES = [
    {
        "recipe": recipe,
        "gpus": [2],
        "data": "synthetic",
        "knobs": {**_LOSS, "GRAD_CKPT": ckpt},
    }
    for recipe in ("recipes_kvq_f16c256_production", "recipes_kvq_f16c256")
    for ckpt in (True, False)
]
