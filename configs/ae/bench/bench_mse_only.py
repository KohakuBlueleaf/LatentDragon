"""KohakuVQ with MSE alone, so the difference to the recipe is the perceptual losses."""

from kohakuengine import use_config
from sensecraft import Term

use_config("../recipes/recipes_kvq_f16c256_production.py")

NAME = "bench_mse_only"
LOSS_CONFIGS = [Term("mse")]
