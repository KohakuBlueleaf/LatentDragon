"""Channels-last against the plain layout on conv recipes, through the bucketed reducer."""

from pathlib import Path

OUT = Path("runs/bench/channels_last_buckets.jsonl")
STEPS = 100
_BASE = {"COMPILE": "model", "GRAD_CKPT": True, "COMPILE_LOSS": True}
CASES = [
    {
        "recipe": recipe,
        "gpus": [2],
        "data": "synthetic",
        "knobs": {**_BASE, "CHANNELS_LAST": layout},
    }
    for recipe in ("recipes_sdvae_f8c4", "recipes_latentmaid_0222_f8c4")
    for layout in (False, True)
]
