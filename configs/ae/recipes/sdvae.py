"""SD-VAE with the taming training setup.

kogine run scripts/train/ae.py --config configs/ae/recipes/sdvae.py
"""

from kohakuengine import use_config

use_config("../data/imagenet.py")
use_config("../precision/fp16.py")
use_config("../train/taming.py")
use_config("../arch/sdvae.py")

NAME = "sdvae"
