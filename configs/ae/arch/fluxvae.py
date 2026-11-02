"""FLUX VAE layout: the SD-VAE layout with 16 latent channels, f8, KL latent."""

ARCH = {
    "domain": "image",
    "encoder": {
        "name": "sdvae",
        "in_channels": 3,
        "out_channels": 32,
        "stages": [[1, 128, 2], [2, 256, 2], [2, 512, 2], [2, 512, 3], [1, 512, 1]],
        "norm": "group32",
        "attn": [0, 0, 0, 1, 0],
    },
    "latent": "kl",
    "decoder": {
        "name": "sdvae",
        "in_channels": 16,
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
