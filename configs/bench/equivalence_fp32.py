"""One step against the research code at fp32, with fp32 loss weights."""

PRECISION = "fp32"
KNOBS = {"LOSS_16BIT": False}
OUT = "runs/experiments/step_gradients_true32.jsonl"
TF32 = False
