"""The production recipe with the critic on from step 0."""

from kohakuengine import use_config

use_config("../recipes/recipes_kvq_f16c256_production.py")

NAME = "bench_production_active"
GAN_SCHEME = {"name": "taming", "disc_start": 0, "disc_weight": 0.5}
