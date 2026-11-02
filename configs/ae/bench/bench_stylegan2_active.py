"""StyleGAN2 losses running from step 0."""

from kohakuengine import use_config

use_config("../recipes/recipes_latentmaid_f8c4_stylegan2.py")

NAME = "bench_stylegan2_active"
GAN_SCHEME = {"name": "stylegan2", "disc_start": 0, "disc_weight": 0.5, "gamma": 1.0}
