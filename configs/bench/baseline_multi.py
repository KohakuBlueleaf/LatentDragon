"""Four GPUs on real data, the same recipes as `baseline_sync.py`."""

from pathlib import Path

OUT = Path("runs/bench/baseline_multi.jsonl")
STEPS = 120
_NAMES = [
    "recipes_kvq_f16c256",
    "recipes_kvq_f16c256_production",
    "recipes_latentmaid_f8c4",
    "recipes_latentmaid_0222_f8c4",
    "recipes_sdvae_f8c4",
    "bench_taming_active",
    "bench_production_active",
    "bench_r3gan_active",
]
CASES = [{"recipe": name, "gpus": [0, 1, 2, 3], "data": "real"} for name in _NAMES]
CASES += [{"recipe": name, "gpus": [0], "data": "real"} for name in _NAMES[:3]]
