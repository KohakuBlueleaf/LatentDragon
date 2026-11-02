"""Building blocks shared by the families: norms, blocks, stages, temporal layers, reach."""

from latentdragon.model.nn.blocks import BLOCK, Block, build_block
from latentdragon.model.nn.layers import NORM, AttnBlock, SwiGLU, resolve_norm
from latentdragon.model.nn.reach import Reach

__all__ = [
    # registries
    "BLOCK",
    "NORM",
    # base classes and records
    "Block",
    "Reach",
    # layers
    "AttnBlock",
    "SwiGLU",
    # builders
    "build_block",
    "resolve_norm",
]
