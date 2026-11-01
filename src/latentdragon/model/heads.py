"""Heads: trunk features to output."""

from collections.abc import Callable

import torch.nn as nn
from optimfactory import mup_init, mup_init_output
from torch import Tensor

from latentdragon_common.registry import Registry

HEAD = Registry("head")


@HEAD.register("regression")
class RegressionHead(nn.Module):
    """Optional norm and activation, then the final projection."""

    needs_pixels = False

    def __init__(
        self,
        dim: int,
        out_dim: int,
        kernel_size: int = 1,
        norm_layer: Callable | None = None,
        act: Callable | None = None,
        conv: type[nn.Conv2d] = nn.Conv2d,
    ):
        super().__init__()
        self.norm = norm_layer(dim) if norm_layer else nn.Identity()
        self.act = act() if act else nn.Identity()
        self.proj = conv(dim, out_dim, kernel_size, 1, kernel_size // 2)
        mup_init(self.parameters())
        mup_init_output(self.proj.weight)

    @property
    def anchor(self) -> nn.Parameter:
        """The final projection weight."""
        return self.proj.weight

    def forward(self, features: Tensor, pixels: Tensor | None = None) -> Tensor:
        """Project trunk features to the output; `pixels` is unused."""
        return self.proj(self.act(self.norm(features)))
