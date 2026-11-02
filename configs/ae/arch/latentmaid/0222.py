"""LatentMaid 0/2/2/2: MLP encoder with two blocks at f2, f4 and f8; SD-VAE decoder.

Encoder stages are `(scale, hidden, mlp, blocks)`; the decoder is the SD-VAE decoder.
"""

ARCH = {
    "domain": "image",
    "encoder": {
        "name": "latentmaid",
        "in_channels": 3,
        "out_channels": 8,
        "stages": [[2, 128, 256, 2], [2, 256, 512, 2], [2, 512, 1024, 2]],
        "norm": "rms",
        "attn": None,
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
