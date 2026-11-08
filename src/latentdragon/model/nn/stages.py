"""Stage parts: stage specs, block stacks, attention slots, conv stages."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from latentdragon.model.nn.blocks import build_block
from latentdragon.model.nn.layers import AttnBlock


@dataclass(frozen=True)
class StageSpec:
    """One shuffle stage; `hidden` 0 or None keeps the incoming channel count."""

    t_scale: int
    scale: int
    hidden: int | None
    mlp: int
    blocks: int

    @classmethod
    def image(cls, cfg: Sequence[int]) -> "StageSpec":
        """From `(scale, hidden, mlp, blocks)`."""
        return cls(1, *cfg)

    @classmethod
    def video(cls, cfg: Sequence[int]) -> "StageSpec":
        """From `(t_scale, scale, hidden, mlp, blocks)`."""
        return cls(*cfg)


def call_block(block: nn.Module, x: Tensor, **frames) -> Tensor:
    """Default per-block step of a stage walk."""
    return block(x, **frames)


def keep_at(keep: Tensor, x: Tensor) -> Tensor:
    """The cell mask `keep` repeated up to `x`'s spatial size, which it must divide, in `x`'s
    dtype so a 16-bit walk stays 16-bit."""
    rows = x.shape[-2] // keep.shape[-2]
    cols = x.shape[-1] // keep.shape[-1]
    return keep.to(x.dtype).repeat_interleave(rows, -2).repeat_interleave(cols, -1)


def masked_step(block: nn.Module, x: Tensor, keep: Tensor) -> Tensor:
    """Per-block step of a masked stage walk: the block's `masked` at `x`'s resolution."""
    return block.masked(x, keep_at(keep, x))


def block_stack(
    block: Any, count: int, dim: int, mlp_dim: int, norm_layer: Callable
) -> nn.ModuleList:
    """`count` same-width blocks."""
    return nn.ModuleList(
        build_block(
            block, in_dim=dim, mlp_dim=mlp_dim, out_dim=dim, norm_layer=norm_layer
        )
        for _ in range(count)
    )


def attn_slots(
    flags: Sequence[int] | None, channels: Sequence[int], norm_layer: Callable
) -> nn.ModuleList:
    """One `AttnBlock` or Identity per stage, so stage indices stay aligned."""
    if not flags:
        return nn.ModuleList(nn.Identity() for _ in channels)
    if len(flags) != len(channels):
        raise ValueError(
            f"attention flags length {len(flags)} must match stage count "
            f"{len(channels)}"
        )
    return nn.ModuleList(
        AttnBlock(dim, norm_layer=norm_layer) if flag else nn.Identity()
        for flag, dim in zip(flags, channels)
    )


class Upsample(nn.Module):
    """Nearest interpolation, then a 3x3 conv."""

    def __init__(self, in_dim: int, out_dim: int, scale: int = 2):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 1, 1)
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        """Upsample, then convolve."""
        return self.conv(F.interpolate(x, scale_factor=self.scale, mode="nearest"))


class Downsample(nn.Module):
    """Stride-2 3x3 conv with bottom-right padding."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, 3, 2, 0)

    def forward(self, x: Tensor) -> Tensor:
        """Pad right and bottom, then strided convolve."""
        return self.conv(F.pad(x, (0, 1, 0, 1), mode="constant", value=0))


def make_sampler(dim: int, scale: int, downsample: bool) -> nn.Module:
    """The stage sampler; only a factor of 2 can go down."""
    match scale, downsample:
        case 1, _:
            return nn.Identity()
        case _, False:
            return Upsample(dim, dim, scale=scale)
        case 2, True:
            return Downsample(dim, dim)
    raise ValueError(f"conv downsampling supports scale 1 or 2, got {scale}")


class ConvStage(nn.Module):
    """Residual blocks, then the sampler."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        blocks: int,
        scale: int = 1,
        norm_layer: Callable | None = None,
        downsample: bool = False,
        block: Any = "res",
    ):
        super().__init__()
        self.blocks = nn.ModuleList(
            build_block(
                block,
                in_dim=out_dim if index else in_dim,
                out_dim=out_dim,
                norm_layer=norm_layer,
            )
            for index in range(blocks)
        )
        self.sample = make_sampler(out_dim, scale, downsample)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the blocks in order, then the sampler."""
        for block in self.blocks:
            x = block(x)
        return self.sample(x)


def product(values: Sequence[int]) -> int:
    """The total scale of a stage stack."""
    total = 1
    for value in values:
        total *= value
    return total
