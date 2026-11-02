"""KohakuVQ with MSE alone, so the difference to the recipe is the perceptual losses."""

from kohakuengine import use_config
from sensecraft.loss import GeneralConfig

use_config("../recipes/recipes_kvq_f16c256_production.py")

NAME = "bench_mse_only"
LOSS_CONFIGS = [GeneralConfig("mse", weight=1.0)]
