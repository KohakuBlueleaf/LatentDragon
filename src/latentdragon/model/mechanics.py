"""In-place model mechanics; neither changes state dict keys."""

from functools import partial

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from latentdragon.model.blocks import Block
from latentdragon.model.layers import AttnBlock


def apply_recompute(model: nn.Module) -> None:
    """Swap every block's `forward` for a checkpointed callable."""
    for module in model.modules():
        if isinstance(module, Block) and "forward" not in vars(module):
            module.forward = partial(checkpoint, module.forward, use_reentrant=False)


def apply_compile(model: nn.Module) -> None:
    """Compile every block and attention block in place."""
    for module in model.modules():
        if isinstance(module, (Block, AttnBlock)):
            module.compile()
