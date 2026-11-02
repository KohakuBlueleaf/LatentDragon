"""Four GPUs, synthetic data: does the data phase cost come from rank CPU threads?"""

from pathlib import Path

OUT = Path("runs/bench/rank_threads.jsonl")
STEPS = 100
CASES = [
    {
        "recipe": "recipes_kvq_f16c256_production",
        "gpus": [0, 1, 2, 3],
        "data": "synthetic",
        "knobs": {"COMPILE": "model", "COMPILE_LOSS": True},
    }
]
