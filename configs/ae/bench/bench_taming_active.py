"""Taming GAN running from step 0, adaptive weight on, stat-free critic."""

from kohakuengine import use_config

use_config("../recipes/recipes_kvq_f16c256.py")

NAME = "bench_taming_active"
GAN_SCHEME = {"name": "taming", "disc_start": 0, "disc_weight": 0.5}
