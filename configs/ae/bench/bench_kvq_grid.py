"""KohakuVQ with a partition x residual grid, hyper mode and IVQ."""

from kohakuengine import use_config

use_config("../recipes/recipes_kvq_f16c256.py")

NAME = "bench_kvq_grid"
ARCH_OVERRIDES = {
    "bottleneck": {
        "name": "vq",
        "quantizer": {
            "name": "kohaku",
            "n_embed": 4096,
            "partitions": 4,
            "rvq_levels": 2,
            "base_dim": 16,
            "partition_mode": "hyper",
            "mlp_projector": True,
            "mlp_hidden_dim": 512,
            "ivq_loss_weight": 1e-3,
        },
    }
}
