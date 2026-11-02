"""The compiled r3gan step with the gate open."""

from pathlib import Path

RECIPE = "bench_r3gan_active"
KNOBS = {"COMPILE": "model", "GRAD_CKPT": True, "COMPILE_LOSS": True}
OUT = Path("runs/bench/op_profile_r3gan.jsonl")
