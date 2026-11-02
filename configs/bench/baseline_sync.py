"""One GPU, synthetic data: the compute side of every recipe and bench config."""

from pathlib import Path

OUT = Path("runs/bench/baseline_synthetic.jsonl")
STEPS = 120
_NAMES = [
    "recipes_kvq_f16c256",
    "recipes_kvq_f16c256_production",
    "recipes_kvq_f16c256_stylegan2",
    "recipes_latentmaid_f8c4",
    "recipes_latentmaid_f8c4_r3gan",
    "recipes_latentmaid_f8c4_stylegan2",
    "recipes_latentmaid_0222_f8c4",
    "recipes_latentmaid_0222_better_dec_f8c4",
    "recipes_latentmaid_0222_f16c64n16",
    "recipes_sdvae_f8c4",
    "bench_kvq_grid",
    "bench_kvq_nested",
    "bench_naive_vq",
    "bench_taming_active",
    "bench_taming_acc",
    "bench_production_active",
    "bench_r3gan_active",
    "bench_stylegan2_active",
    "bench_video",
    "bench_mse_only",
]
CASES = [{"recipe": name, "gpus": [0], "data": "synthetic"} for name in _NAMES]
