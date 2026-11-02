"""R3GAN with R1 and R2 running from step 0."""

from kohakuengine import use_config

use_config("../recipes/recipes_latentmaid_f8c4_r3gan.py")

NAME = "bench_r3gan_active"
GAN_SCHEME = {"name": "r3gan", "disc_start": 0, "disc_weight": 0.5, "gamma": 1.0}
