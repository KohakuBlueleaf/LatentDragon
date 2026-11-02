"""In-place model mechanics; none of them changes state dict keys."""

from functools import partial

import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from latentdragon.model.blocks import Block
from latentdragon.model.compile import compile_wrapper


def _plain(forward):
    """The undecorated function behind a `compile_wrapper` forward."""
    return getattr(forward, "__wrapped__", forward)


def recomputed(block: Block):
    """`block.forward` checkpointed, compiled as one graph under the `module` scope.

    The block's own compiled forward would hide its ops from the checkpoint, so the plain
    forward goes in and the recompute fuses into the backward kernels.
    """
    return compile_wrapper(
        partial(
            checkpoint, partial(_plain(type(block).forward), block), use_reentrant=False
        ),
        scope="module",
    )


def apply_recompute(model: nn.Module) -> None:
    """Swap every block's `forward` for a checkpointed callable."""
    for module in model.modules():
        if isinstance(module, Block) and "forward" not in vars(module):
            module.forward = recomputed(module)
