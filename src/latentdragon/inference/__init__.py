"""Decoding latents too large to run at once: tiled, staged and streaming strategies."""

import torch
from torch import Tensor

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
def decode(model: Autoencoder, z: Tensor, spec: str | dict = "full") -> Tensor:
    """Decode bottleneck latents `z` with the strategy named by `spec`."""
    strategy = build(spec, DECODE)
    strategy.check(model)
    return strategy(model, z)


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
