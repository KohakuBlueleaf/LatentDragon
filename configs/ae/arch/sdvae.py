"""SD-VAE layout: f8, 4 latent channels, KL, group norm, attention at the lowest scale."""

ARCH = {
    "domain": "image",
    "image_dim": 3,
    "latent_dim": 4,
    "encoder": "conv",
    "encoder_blocks": [[1, 128, 2], [2, 256, 2], [2, 512, 2], [2, 512, 3], [1, 512, 1]],
    "encoder_attn": [0, 0, 0, 1, 0],
    "encoder_norm": "group32",
    "bottleneck": "kl",
    "trunk": "conv",
    "trunk_blocks": [[1, 512, 1], [2, 512, 4], [2, 512, 3], [2, 256, 3], [1, 128, 3]],
    "trunk_attn": [0, 1, 0, 0, 0],
    "trunk_norm": "group32",
    "head": "regression",
}
