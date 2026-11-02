"""Decoding latents too large to run at once: tiled, staged and streaming strategies."""

import torch

from latentdragon.inference.strategy import (
    DECODE,
    DecodeStrategy,
    FullDecode,
    StagedTiledDecode,
    TiledDecode,
)
from latentdragon.inference.streaming import (
    StreamingDecode,
    StreamingTiledDecode,
    stream_encode,
)
from latentdragon.model import Autoencoder
from latentdragon_common.registry import build


@torch.no_grad()
def decode(model: Autoencoder, latent: dict, spec: str | dict = "full") -> dict:
    """Decode a live or packed latent with the strategy named by `spec`.

    The latent def reads the latent into the decoder's slot; the strategy decodes that
    tensor and the result comes back under the head's output slot.
    """
    strategy = build(spec, DECODE)
    strategy.check(model)
    z = model.latent.deserialize(latent)[model.decoder.input]
    return {model.head.output: strategy(model, z)}


__all__ = [
    # registries
    "DECODE",
    # base classes
    "DecodeStrategy",
    # concrete classes
    "FullDecode",
    "StagedTiledDecode",
    "StreamingDecode",
    "StreamingTiledDecode",
    "TiledDecode",
    # functions
    "decode",
    "stream_encode",
]
