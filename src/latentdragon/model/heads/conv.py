"""Heads: the output latent to the signal."""

from collections.abc import Callable

import torch.nn as nn
from optimfactory import mup_init, mup_init_output
from torch import Tensor

from latentdragon.model.core.contracts import HEAD, TensorPart
from latentdragon.model.nn.layers import resolve_norm
from latentdragon.model.nn.temporal import SpatialConv2d

ACTS = {None: None, "silu": nn.SiLU, "mish": nn.Mish}


@HEAD.register("conv")
class ConvHead(TensorPart):
    """Optional norm and activation, then a `kernel_size` projection to `out_channels`.

    `video` applies the projection per frame of a `(B, T, C, H, W)` slot.
    """

    objective = "reconstruction"
    needs_pixels = False

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 3,
        kernel_size: int = 1,
        norm: str | Callable | None = None,
        act: str | None = None,
        video: bool = False,
        input: str = "y",
        output: str = "x",
    ):
        super().__init__(input, output)
        conv = SpatialConv2d if video else nn.Conv2d
        self.norm = resolve_norm(norm)(in_channels) if norm else nn.Identity()
        self.act = ACTS[act]() if act else nn.Identity()
        self.proj = conv(in_channels, out_channels, kernel_size, 1, kernel_size // 2)
        mup_init(self.parameters())
        mup_init_output(self.proj.weight)

    @property
    def last_layer(self) -> nn.Module:
        """The final projection."""
        return self.proj

    @property
    def reach(self) -> int:
        """Output pixels read on each side."""
        return self.proj.kernel_size[0] // 2

    def tensor(self, features: Tensor) -> Tensor:
        """Project features to the output."""
        return self.proj(self.act(self.norm(features)))
