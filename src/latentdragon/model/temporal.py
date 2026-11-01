"""Frame-wise convs, the causal temporal conv and the temporal fusers."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class SpatialConv2d(nn.Conv2d):
    """Conv2d applied to each frame of `(B, T, C, H, W)`."""

    def forward(self, x: Tensor) -> Tensor:
        """Fold frames into the batch, convolve, and unfold."""
        batch, frames = x.shape[:2]
        return super().forward(x.flatten(0, 1)).unflatten(0, (batch, frames))


class SpatialDWConv(SpatialConv2d):
    """Frame-wise depthwise conv."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__(
            dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim
        )


class CausalTemporalConv(nn.Conv1d):
    """Causal full-rank Conv1d over frames; returns the output and new cache."""

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__(dim, dim, kernel_size=kernel_size, padding=0)
        self.context = kernel_size - 1

    def forward(
        self,
        x: Tensor,
        has_keyframe: bool = True,
        causal_cache: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Convolve `(B, T, C, H, W)` over time; returns the output and the new cache."""
        if has_keyframe and causal_cache is not None:
            raise ValueError("a keyframe chunk starts a stream; it takes no cache")
        batch, frames, _, height, width = x.shape

        # Per-pixel frame lines, prefixed by the cache; keep the tail as new cache
        # (B, T, C, H, W) -> (B*H*W, C, T)
        line = x.permute(0, 3, 4, 2, 1).flatten(0, 2)
        if causal_cache is not None:
            line = torch.cat([causal_cache, line], dim=-1)
        cache = line[:, :, -self.context :]
        if has_keyframe and frames == 1:
            return x, cache

        # A keyframe keeps its slot, so the conv emits one frame fewer.
        wanted = frames - 1 if has_keyframe else frames
        padded = F.pad(line, (wanted + self.context - line.size(-1), 0))
        out = super().forward(padded).unflatten(0, (batch, height, width))
        out = out.permute(0, 4, 3, 1, 2)
        if has_keyframe:
            out = torch.cat([x[:, :1], out], dim=1)
        return out, cache


class IdentityFuser(nn.Module):
    """A stage with no temporal scaling; same call shape as the fusers."""

    def forward(self, x: Tensor, has_keyframe: bool = True) -> Tensor:
        """Return `x` unchanged."""
        return x


class TemporalDownsample(nn.Module):
    """Concat `factor` frames along channels, project back; a keyframe passes through."""

    def __init__(self, dim: int, factor: int):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(factor * dim, dim)

    def forward(self, x: Tensor, has_keyframe: bool = True) -> Tensor:
        """Merge every `factor` frames into one; `(B, n*factor, C, H, W)` to `(B, n, C, H, W)`."""
        rest = x[:, 1:] if has_keyframe else x
        if rest.size(1) == 0:
            return x
        # (B, n*factor, C, H, W) -> (B, n, H, W, k*C)
        rest = rest.unflatten(1, (-1, self.factor))
        rest = rest.permute(0, 1, 4, 5, 2, 3).flatten(4, 5)
        rest = self.proj(rest).permute(0, 1, 4, 2, 3)
        return torch.cat([x[:, :1], rest], dim=1) if has_keyframe else rest


class TemporalUpsample(nn.Module):
    """Project each frame to `factor` frames; a keyframe passes through."""

    def __init__(self, dim: int, factor: int):
        super().__init__()
        self.factor = factor
        self.proj = nn.Linear(dim, factor * dim)

    def forward(self, x: Tensor, has_keyframe: bool = True) -> Tensor:
        """Split every frame into `factor`; `(B, n, C, H, W)` to `(B, n*factor, C, H, W)`."""
        rest = x[:, 1:] if has_keyframe else x
        if rest.size(1) == 0:
            return x
        # (B, n, C, H, W) -> (B, n, H, W, k, C) -> (B, n*factor, C, H, W)
        rest = self.proj(rest.permute(0, 1, 3, 4, 2))
        rest = rest.unflatten(4, (self.factor, -1))
        rest = rest.permute(0, 1, 4, 5, 2, 3).flatten(1, 2)
        return torch.cat([x[:, :1], rest], dim=1) if has_keyframe else rest


def make_fuser(fuser: type[nn.Module], dim: int, t_scale: int) -> nn.Module:
    """`fuser` for a temporally scaling stage, otherwise the identity."""
    return fuser(dim, t_scale) if t_scale > 1 else IdentityFuser()
