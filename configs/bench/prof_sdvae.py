"""The compiled sdvae step."""

from pathlib import Path

RECIPE = "recipes_sdvae_f8c4"
KNOBS = {"COMPILE": "model", "GRAD_CKPT": True, "COMPILE_LOSS": True}
OUT = Path("runs/bench/op_profile_sdvae.jsonl")
