"""Encoders, trunks, heads and the autoencoder that composes them."""

from latentdragon.model.autoencoder import Autoencoder
from latentdragon.model.blocks import BLOCK, Block, build_block
from latentdragon.model.encoders import (
    ENCODER,
    ConvEncoder,
    MLPEncoder,
    VideoEncoder,
)
from latentdragon.model.heads import HEAD, RegressionHead
from latentdragon.model.layers import NORM, AttnBlock, SwiGLU, resolve_norm
from latentdragon.model.mechanics import apply_compile, apply_recompute
from latentdragon.model.presets import (
    PRESETS,
    AEArchConfig,
    build_autoencoder,
    get_preset,
)
from latentdragon.model.reach import Reach
from latentdragon.model.trunks import TRUNK, ConvTrunk, MLPTrunk, VideoTrunk

__all__ = [
    # registries
    "BLOCK",
    "ENCODER",
    "HEAD",
    "NORM",
    "PRESETS",
    "TRUNK",
    # base classes and configs
    "AEArchConfig",
    "Block",
    "Reach",
    # concrete classes
    "AttnBlock",
    "Autoencoder",
    "ConvEncoder",
    "ConvTrunk",
    "MLPEncoder",
    "MLPTrunk",
    "RegressionHead",
    "SwiGLU",
    "VideoEncoder",
    "VideoTrunk",
    # builders and functions
    "apply_compile",
    "apply_recompute",
    "build_autoencoder",
    "build_block",
    "get_preset",
    "resolve_norm",
]
