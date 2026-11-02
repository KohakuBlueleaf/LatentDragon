"""Short fresh run of production_active; set PRECISION and NAME with `--set`."""

from kohakuengine import use_config

use_config("../../ae/bench/bench_production_active.py")
use_config("_short.py")
