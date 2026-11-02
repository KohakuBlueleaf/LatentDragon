"""Compile scope against recompute, one GPU, synthetic data, current tree."""

from pathlib import Path

OUT = Path("runs/bench/compile_matrix.jsonl")
STEPS = 100
WARM_STEPS = 6
_RECIPES = [
    "recipes_kvq_f16c256_production",
    "recipes_kvq_f16c256",
    "recipes_latentmaid_0222_f8c4",
    "recipes_sdvae_f8c4",
]
_ARMS = [
    {"COMPILE": "off", "GRAD_CKPT": True},
    {"COMPILE": "off", "GRAD_CKPT": False},
    {"COMPILE": "function", "GRAD_CKPT": True},
    {"COMPILE": "function", "GRAD_CKPT": False},
    {"COMPILE": "module", "GRAD_CKPT": True},
    {"COMPILE": "module", "GRAD_CKPT": False},
    {"COMPILE": "model", "GRAD_CKPT": True},
    {"COMPILE": "model", "GRAD_CKPT": False},
]
CASES = [
    {"recipe": recipe, "gpus": [0], "data": "synthetic", "knobs": arm}
    for recipe in _RECIPES
    for arm in _ARMS
]
