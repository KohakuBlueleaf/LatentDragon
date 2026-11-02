"""Short real run of the kvq recipe under the bf16-true policy."""

import os

from kohakuengine import use_config

use_config("../../ae/recipes/recipes_kvq_f16c256.py")

NAME = "policy_bf16_true"
RUN_ID = "run"
RUNS_DIR = "runs/policy"
PRECISION = "bf16-true"
OPTIMIZER = "adam16"
TRAIN_SOURCES = [
    {"name": "image_folder", "root": os.environ["LATENTDRAGON_TRAIN_ROOTS"]}
]
COMPILE = "model"
COMPILE_LOSS = True
GRAD_CKPT = False
DEVICES = [0]
MAX_STEPS = 3000
EPOCHS = 0
LOG_EVERY = 100
METRICS_EVERY = 100
SAMPLE_EVERY = 0
VAL_EVERY = 0
CKPT_EVERY = 100000
WANDB_PROJECT = None
