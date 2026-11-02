"""Short fresh run of sdvae_f8c4; set PRECISION and NAME with `--set`."""

from kohakuengine import use_config

use_config("../../ae/recipes/recipes_sdvae_f8c4.py")
use_config("_short.py")
