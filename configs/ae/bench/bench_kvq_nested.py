"""Partition-level nested dropout with jitter around KohakuVQ."""

from kohakuengine import use_config

use_config("../recipes/recipes_kvq_f16c256.py")

NAME = "bench_kvq_nested"
ARCH_OVERRIDES = {
    "bottleneck": {
        "name": "nested",
        "unit": "partition",
        "latent_chs": [2, 4],
        "jitter": 1,
        "inner": {
            "name": "vq",
            "quantizer": {
                "name": "kohaku",
                "n_embed": 4096,
                "partitions": 8,
                "base_dim": 16,
                "partition_mode": "hyper",
                "mlp_projector": True,
                "mlp_hidden_dim": 512,
            },
        },
    }
}
