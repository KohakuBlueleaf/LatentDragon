"""Short fresh run of r3gan_active; set PRECISION and NAME with `--set`."""

from kohakuengine import use_config

use_config("../../ae/bench/bench_r3gan_active.py")
use_config("_short.py")
