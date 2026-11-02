"""The compiled latentmaid_f8c4 step."""

from pathlib import Path

RECIPE = "recipes_latentmaid_f8c4"
KNOBS = {"COMPILE": "model", "GRAD_CKPT": True, "COMPILE_LOSS": True}
OUT = Path("runs/bench/op_profile_latentmaid_f8c4.jsonl")
