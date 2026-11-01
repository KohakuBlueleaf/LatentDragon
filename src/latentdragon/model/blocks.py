"""Residual blocks; `build_block` is the one construction point."""

from collections.abc import Callable
from typing import Any

import torch.nn as nn
from torch import Tensor

from latentdragon.model.layers import RMSNorm, SwiGLU
from latentdragon.model.temporal import CausalTemporalConv, SpatialDWConv
from latentdragon_common.registry import Registry, build

BLOCK = Registry("block")


def build_block(spec: Any, **site) -> nn.Module:
    """A block from a registry name, dotted path, `{name: ..}` dict or factory."""
    return build(spec, BLOCK, **site)


class Block(nn.Module):
    """Marks the modules `apply_recompute` and `apply_compile` act on.

    `spatial_reach` is the pixels one block reads on each side at its resolution, None
    when unknown. `temporal_reach` is the past frames it reads.
    """

    spatial_reach: int | None = None
    temporal_reach: int = 0


def _channel_mlp(in_dim: int, mlp_dim: int, out_dim: int) -> nn.Module:
    """A SwiGLU channel mixer, or a plain linear when `mlp_dim` is 0."""
    return (
        SwiGLU(in_dim, mlp_dim, out_dim) if mlp_dim > 0 else nn.Linear(in_dim, out_dim)
    )


@BLOCK.register("mlp")
class BasicBlock(Block):
    """Norm, depthwise 3x3, channel MLP, residual."""

    spatial_reach = 1

    def __init__(
        self, in_dim: int, mlp_dim: int, out_dim: int, norm_layer: Callable = RMSNorm
    ):
        super().__init__()
        self.norm = norm_layer(in_dim)
        self.mixer = nn.Conv2d(in_dim, in_dim, 3, 1, 1, groups=in_dim)
        self.swiglu = _channel_mlp(in_dim, mlp_dim, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        """Spatial mix, then channel MLP on channels-last, added to `x`."""
        mixed = self.mixer(self.norm(x)).transpose(1, -1)
        return x + self.swiglu(mixed).transpose(1, -1)


@BLOCK.register("res")
class BasicResBlock(Block):
    """SD-style block: two rounds of norm, activation, 3x3 conv, with a skip."""

    spatial_reach = 2

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        norm_layer: Callable = RMSNorm,
        act: Callable = nn.SiLU,
    ):
        super().__init__()
        self.norm1 = norm_layer(in_dim)
        self.nonlin1 = act()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.norm2 = norm_layer(out_dim)
        self.nonlin2 = act()
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, 1, 1)
        self.skip = (
            nn.Conv2d(in_dim, out_dim, 1) if in_dim != out_dim else nn.Identity()
        )

    def forward(self, x: Tensor) -> Tensor:
        """Two norm-act-conv rounds added to the (projected) input."""
        hidden = self.conv1(self.nonlin1(self.norm1(x)))
        hidden = self.conv2(self.nonlin2(self.norm2(hidden)))
        return self.skip(x) + hidden


@BLOCK.register("mlp3d")
class BasicBlock3D(Block):
    """`BasicBlock` on `(B, T, C, H, W)` with a causal temporal conv after the mixer."""

    spatial_reach = 1
    temporal_reach = 2

    def __init__(
        self, in_dim: int, mlp_dim: int, out_dim: int, norm_layer: Callable = RMSNorm
    ):
        super().__init__()
        self.norm = norm_layer(in_dim)
        self.mixer = SpatialDWConv(in_dim)
        self.tmixer = CausalTemporalConv(in_dim)
        self.swiglu = _channel_mlp(in_dim, mlp_dim, out_dim)

    def stream(
        self, x: Tensor, has_keyframe: bool = True, causal_cache: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """One chunk with its cached frames; returns the output and new cache."""
        batch, frames = x.shape[:2]

        # Frame-wise norm and spatial mix, then the causal temporal mix
        mixed = self.mixer(self.norm(x.flatten(0, 1)).unflatten(0, (batch, frames)))
        mixed, cache = self.tmixer(mixed, has_keyframe, causal_cache)

        # Channel MLP on channels-last, residual
        return x + self.swiglu(mixed.transpose(2, -1)).transpose(2, -1), cache

    def forward(self, x: Tensor, has_keyframe: bool = True) -> Tensor:
        """The non-streaming pass: `stream` without a cache, output only."""
        return self.stream(x, has_keyframe)[0]
