"""LatentMaid 0/2/2/2: MLP encoder with two blocks at f2, f4 and f8; SD-VAE decoder.

Encoder stages are `(scale, hidden, mlp, blocks)`; the decoder is the SD-VAE decoder.
"""

ARCH = {
    "domain": "image",
    "image_dim": 3,
    "latent_dim": 4,
    "encoder": "mlp",
    "encoder_blocks": [[2, 128, 256, 2], [2, 256, 512, 2], [2, 512, 1024, 2]],
    "encoder_attn": None,
    "encoder_norm": "rms",
    "bottleneck": "kl",
    "trunk": "conv",
    "trunk_blocks": [[1, 512, 1], [2, 512, 4], [2, 512, 3], [2, 256, 3], [1, 128, 3]],
    "trunk_attn": [0, 1, 0, 0, 0],
    "trunk_norm": "group32",
    "head": "regression",
}
