"""`Reach`: how far the decode of one latent position looks at its neighbors."""

from collections.abc import Iterable
from dataclasses import dataclass

import torch.nn as nn

from latentdragon.model.nn.layers import AttnBlock

# These read every position of the input, so no finite overlap is exact
NONLOCAL = (AttnBlock, nn.GroupNorm)


@dataclass(frozen=True)
class Reach:
    """Grid cells one output position depends on, past its own cell.

    `spatial` counts cells on each side, None when the path holds a global op.
    `temporal` counts past frames; decoding never looks at future frames.
    """

    spatial: int | None
    temporal: int = 0


def is_local(modules: Iterable[nn.Module]) -> bool:
    """False when any module, or anything inside it, is a global op."""
    return not any(
        isinstance(part, NONLOCAL) for module in modules for part in module.modules()
    )


def declared(blocks: Iterable[nn.Module]) -> bool:
    """True when every block declares its spatial reach."""
    return all(getattr(block, "spatial_reach", None) is not None for block in blocks)
