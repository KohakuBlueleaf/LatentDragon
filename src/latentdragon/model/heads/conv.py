"""Heads: the output latent to the signal."""

from collections.abc import Callable

import torch
import torch.nn as nn
from optimfactory import mup_init, mup_init_output
from torch import Tensor

from latentdragon.model.core.contracts import HEAD, TensorPart
from latentdragon.model.nn.layers import resolve_norm
from latentdragon.model.nn.temporal import SpatialConv2d

ACTS = {None: None, "silu": nn.SiLU, "mish": nn.Mish}


class Standardized:
    """A conv whose weight rows are zero-mean and unit-norm at every call.

    Each output channel reads a fixed-scale direction of its input, with no learnable gain, so
    the head's scale cannot drift and a bounded target bounds what it reads.
    """

    def _conv_forward(self, x: Tensor, weight: Tensor, bias):
        rows = weight - weight.mean(dim=(1, 2, 3), keepdim=True)
        norm = rows.flatten(1).norm(dim=1).clamp_min(1e-6).view(-1, 1, 1, 1)
        return super()._conv_forward(x, rows / norm, bias)


class StandardizedConv2d(Standardized, nn.Conv2d):
    """`nn.Conv2d` with standardized weight rows."""


class StandardizedSpatialConv2d(Standardized, SpatialConv2d):
    """`SpatialConv2d` with standardized weight rows."""


@HEAD.register("conv")
class ConvHead(TensorPart):
    """Optional norm and activation, then a `kernel_size` projection to `out_channels`.

    `video` applies the projection per frame of a `(B, T, C, H, W)` slot. `standardize`
    makes every projection row zero-mean and unit-norm at each call, fixing the head's scale.
    `zero_init` starts the projection's weight and bias at zero. `fp32` runs the head outside
    autocast on an fp32 input, and keeps its weights fp32 under the `-true` precisions, so
    the weight gradient's sum over every output position never lands in 16 bits.
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
        standardize: bool = False,
        zero_init: bool = False,
        fp32: bool = False,
        input: str = "y",
        output: str = "x",
    ):
        super().__init__(input, output)
        self.fp32 = fp32
        if standardize and zero_init:
            raise ValueError("a standardized row needs a direction; zero_init has none")
        conv = {
            (False, False): nn.Conv2d,
            (True, False): SpatialConv2d,
            (False, True): StandardizedConv2d,
            (True, True): StandardizedSpatialConv2d,
        }[video, standardize]
        self.norm = resolve_norm(norm)(in_channels) if norm else nn.Identity()
        self.act = ACTS[act]() if act else nn.Identity()
        self.proj = conv(in_channels, out_channels, kernel_size, 1, kernel_size // 2)
        mup_init(self.parameters())
        mup_init_output(self.proj.weight)
        if zero_init:
            nn.init.zeros_(self.proj.weight)
            nn.init.zeros_(self.proj.bias)

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
        if not self.fp32:
            return self.proj(self.act(self.norm(features)))
        with torch.autocast(features.device.type, enabled=False):
            return self.proj(self.act(self.norm(features.float())))
