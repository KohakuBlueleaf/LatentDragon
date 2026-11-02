"""Shared knobs of a short real run from fresh weights on one GPU."""

import os

TRAIN_SOURCES = [
    {"name": "image_folder", "root": os.environ["LATENTDRAGON_TRAIN_ROOTS"]}
]
RUN_ID = "run"
RUNS_DIR = "runs/fresh"
COMPILE = "model"
COMPILE_LOSS = True
DEVICES = [0]
MAX_STEPS = 1500
EPOCHS = 0
LOG_EVERY = 100
METRICS_EVERY = 50
SAMPLE_EVERY = 0
VAL_EVERY = 0
CKPT_EVERY = 100000
WANDB_PROJECT = None
