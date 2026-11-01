"""Stage two: hold encoder, VQ and trunk, retrain the head on a stage-one checkpoint.

Point $LATENTDRAGON_STAGE1 at a `checkpoints/step_XXXXXXXX` folder of a stage-one run.

    kogine run scripts/train/ae.py --config configs/ae/recipes/recipes_kvq_f16c256_stage2.py
"""

import os

from kohakuengine import use_config

use_config("recipes_kvq_f16c256.py")

NAME = "kvq_f16c256_stage2"
INIT_FROM = os.environ.get("LATENTDRAGON_STAGE1", "stage1")
FREEZE = ["encoder", "bottleneck", "trunk"]
GRAD_CKPT = False
