"""Norms, SwiGLU and residual attention shared by every block family."""

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from latentdragon.model.runtime.compile import compile_wrapper
from latentdragon_common.registry import Registry

NORM = Registry("norm")


def resolve_norm(choice: str | Callable) -> Callable:
    """A norm factory from a registry name, a dotted path or a factory."""
    return NORM.get(choice) if isinstance(choice, str) else choice


@NORM.register("rms")
class RMSNorm(nn.Module):
    """RMS norm over the channel axis at dim 1."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @compile_wrapper
    def forward(self, x):
        """Normalize with channels moved last, then moved back."""
        x = x.transpose(1, -1)
        x = F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        return x.transpose(1, -1)


@NORM.register("layer")
class LayerNorm(nn.Module):
    """Layer norm over the channel axis at dim 1."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    @compile_wrapper
    def forward(self, x):
        """Normalize with channels moved last, then moved back."""
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.transpose(1, -1)


@NORM.register("group32")
class GroupNorm32(nn.GroupNorm):
    """Group norm with 32 groups."""

    def __init__(self, dim: int):
        super().__init__(32, dim, eps=1e-6)


class SwiGLU(nn.Module):
    """Gated SiLU MLP on channels-last input."""

    def __init__(self, in_dim: int, mlp_dim: int, out_dim: int):
        super().__init__()
        self.inp = nn.Linear(in_dim, mlp_dim * 2)
        self.out = nn.Linear(mlp_dim, out_dim)

    @compile_wrapper
    def forward(self, x):
        """`silu(a) * gate` of the split input projection, then the output projection."""
        x, gate = self.inp(x).chunk(2, dim=-1)
        return self.out(F.silu(x) * gate)


class AttnBlock(nn.Module):
    """Residual self-attention over the spatial positions of `(B, C, H, W)`."""

    def __init__(self, dim: int, norm_layer: Callable = RMSNorm, num_heads: int = 1):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} not divisible by num_heads {num_heads}")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.norm = norm_layer(dim)
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=True)
        self.proj_out = nn.Conv2d(dim, dim, kernel_size=1, bias=True)

    def forward(self, x):
        """Attend over the H*W positions and add the projected result to `x`."""
        batch, channels, height, width = x.shape

        # Project to q, k, v with positions as the sequence
        qkv = self.qkv(self.norm(x))
        qkv = qkv.reshape(batch, 3, self.num_heads, self.head_dim, height * width)
        # (B, heads, head_dim, N) -> (B, heads, N, head_dim)
        query, key, value = (
            part.transpose(-1, -2).contiguous() for part in qkv.unbind(dim=1)
        )

        out = F.scaled_dot_product_attention(query, key, value)
        out = out.transpose(-1, -2).reshape(batch, channels, height, width)
        return x + self.proj_out(out)
