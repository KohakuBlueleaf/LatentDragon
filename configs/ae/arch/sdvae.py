"""SD-VAE layout: f8, 4 latent channels, KL, group norm, attention at the lowest scale."""

ARCH = {
    "domain": "image",
    "encoder": {
        "name": "sdvae",
        "in_channels": 3,
        "out_channels": 8,
        "stages": [[1, 128, 2], [2, 256, 2], [2, 512, 2], [2, 512, 3], [1, 512, 1]],
        "norm": "group32",
        "attn": [0, 0, 0, 1, 0],
    },
    "latent": "kl",
    "decoder": {
        "name": "sdvae",
        "in_channels": 4,
        "stages": [[1, 512, 1], [2, 512, 4], [2, 512, 3], [2, 256, 3], [1, 128, 3]],
        "norm": "group32",
        "attn": [0, 1, 0, 0, 0],
    },
    "head": {
        "name": "conv",
        "in_channels": 128,
        "out_channels": 3,
        "kernel_size": 3,
        "norm": "group32",
        "act": "mish",  # KohakuLatentMaid's decoder ends in Mish; SD-VAE uses "silu"
    },
}
