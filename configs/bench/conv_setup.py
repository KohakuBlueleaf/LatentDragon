"""Conv recipes: cuDNN benchmark and channels-last, with and without checkpoints."""

from pathlib import Path

OUT = Path("runs/bench/conv_setup.jsonl")
STEPS = 100
_BASE = {"COMPILE": "model", "COMPILE_LOSS": True}
_ARMS = [
    {"GRAD_CKPT": False, "CHANNELS_LAST": False},
    {"GRAD_CKPT": False, "CHANNELS_LAST": True},
    {"GRAD_CKPT": False, "CHANNELS_LAST": True, "CUDNN_BENCHMARK": True},
    {"GRAD_CKPT": True, "CHANNELS_LAST": True, "CUDNN_BENCHMARK": True},
]
CASES = [
    {"recipe": recipe, "gpus": [0], "data": "synthetic", "knobs": {**_BASE, **arm}}
    for recipe in ("recipes_sdvae_f8c4", "recipes_latentmaid_0222_f8c4")
    for arm in _ARMS
]
