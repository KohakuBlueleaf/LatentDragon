"""Every recipe on one GPU at the fastest setup, with and without gradient checkpoints."""

from pathlib import Path

OUT = Path("runs/bench/final_single.jsonl")
STEPS = 100
_FAST = {"COMPILE": "model", "COMPILE_LOSS": True}
_RECIPES = [
    "recipes_kvq_f16c256_production",
    "recipes_kvq_f16c256",
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
CASES = [
    {
        "recipe": recipe,
        "gpus": [0],
        "data": "synthetic",
        "knobs": {**_FAST, "GRAD_CKPT": ckpt},
    }
    for recipe in _RECIPES
    for ckpt in (True, False)
]
