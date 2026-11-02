"""A plain table codebook with residual levels."""

from kohakuengine import use_config

use_config("../recipes/recipes_kvq_f16c256.py")

NAME = "bench_naive_vq"
ARCH_OVERRIDES = {
    "bottleneck": {
        "name": "vq",
        "quantizer": {"name": "naive", "n_embed": 16384, "rvq_levels": 2},
    }
}
