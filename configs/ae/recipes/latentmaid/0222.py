"""LatentMaid 0/2/2/2 with the taming training setup.

kogine run scripts/train/ae.py --config configs/ae/recipes/latentmaid/0222.py
"""

from kohakuengine import use_config

use_config("../../data/imagenet.py")
use_config("../../precision/fp16.py")
use_config("../../train/taming.py")
use_config("../../arch/latentmaid/0222.py")

NAME = "latentmaid_0222"
