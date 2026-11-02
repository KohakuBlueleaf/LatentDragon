"""What is left after the scope matrix: conv recipes, compiled losses, layout, precision."""

from pathlib import Path

OUT = Path("runs/bench/compile_round2.jsonl")
STEPS = 100
_MODEL = {"COMPILE": "model", "GRAD_CKPT": True}
_NO_CKPT = {"COMPILE": "model", "GRAD_CKPT": False}
_LOSS = {**_MODEL, "COMPILE_LOSS": True}
_ARMS = [
    {"COMPILE": "module", "GRAD_CKPT": True},
    _LOSS,
    {**_LOSS, "LOSS_16BIT": False},
    {**_LOSS, "COMPILE_TUNE": True},
    {**_LOSS, "CHANNELS_LAST": True},
    {**_LOSS, "PRECISION": "fp16"},
    {**_LOSS, "PRECISION": "bf16-true", "OPTIMIZER": "adam16"},
    {**_LOSS, "PRECISION": "fp16-true", "OPTIMIZER": "adam16"},
    {**_LOSS, "REDUCER": "none"},
    {**_NO_CKPT, "COMPILE_LOSS": True},
]
_RECIPES = [
    "recipes_kvq_f16c256_production",
    "recipes_kvq_f16c256",
    "recipes_latentmaid_0222_f8c4",
    "recipes_sdvae_f8c4",
]
CASES = [
    {"recipe": recipe, "gpus": [0], "data": "synthetic", "knobs": arm}
    for recipe in _RECIPES
    for arm in _ARMS
]
