"""Norms, unit-row layers, SwiGLU and residual attention shared by every block family."""

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
    """RMS norm over the channel axis at dim 1; `affine=False` drops the gain."""

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    @compile_wrapper
    def forward(self, x):
        """Normalize with channels moved last, then moved back."""
        x = x.transpose(1, -1)
        x = F.rms_norm(x, (x.shape[-1],), self.weight, self.eps)
        return x.transpose(1, -1)


@NORM.register("layer")
class LayerNorm(nn.Module):
    """Layer norm over the channel axis at dim 1; `affine=False` drops the gain and shift."""

    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None
        self.bias = nn.Parameter(torch.zeros(dim)) if affine else None

    @compile_wrapper
    def forward(self, x):
        """Normalize with channels moved last, then moved back."""
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (x.shape[-1],), self.weight, self.bias, self.eps)
        return x.transpose(1, -1)


@NORM.register("group32")
class GroupNorm32(nn.GroupNorm):
    """Group norm with 32 groups; `affine=False` drops the gain and shift."""

    def __init__(self, dim: int, affine: bool = True):
        super().__init__(32, dim, eps=1e-6, affine=affine)


def unit_rows(weight: torch.Tensor, floor: float = 1e-12) -> torch.Tensor:
    """`weight` with each output row (dim 0, over every other dim) at unit L2 norm.

    The norm is taken in at least fp32 and the result cast back, so a 16-bit weight keeps exact
    rows. `floor` only guards an all-zero row; an added epsilon would make the output depend on
    the raw scale.
    """
    flat = weight.to(torch.promote_types(weight.dtype, torch.float32)).flatten(1)
    unit = flat / flat.norm(dim=1, keepdim=True).clamp_min(floor)
    return unit.view_as(weight).to(weight.dtype)


class UnitRows:
    """Mixin: the layer applies its weight with unit-norm rows, times an optional per-row gain.

    The raw weight's scale never reaches the output, so Adam's growth of it only shrinks later
    steps; with `gain`, that gain is the only scale the layer owns. The bias is added as usual.
    """

    def _add_gain(self, gain: bool) -> None:
        self.gain = nn.Parameter(torch.ones(self.weight.shape[0])) if gain else None

    def unit_weight(self) -> torch.Tensor:
        """The weight the layer applies."""
        weight = unit_rows(self.weight)
        if self.gain is None:
            return weight
        return weight * self.gain.to(weight.dtype).view(-1, *[1] * (weight.dim() - 1))


class UnitRowLinear(UnitRows, nn.Linear):
    """`nn.Linear` with unit-norm rows and an optional per-row gain."""

    def __init__(
        self, in_dim: int, out_dim: int, bias: bool = True, gain: bool = False
    ):
        super().__init__(in_dim, out_dim, bias)
        self._add_gain(gain)

    def forward(self, x):
        return F.linear(x, self.unit_weight(), self.bias)


class UnitRowConv2d(UnitRows, nn.Conv2d):
    """`nn.Conv2d` with unit-norm rows (each output channel's filter) and an optional gain."""

    def __init__(self, *args, gain: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        self._add_gain(gain)

    def _conv_forward(self, x, weight, bias):
        return super()._conv_forward(x, self.unit_weight(), bias)


class SwiGLU(nn.Module):
    """Gated SiLU MLP on channels-last input.

    `unit_rows` gives both projections unit-norm rows and the output projection, the one that
    writes, a per-channel gain.
    """

    def __init__(
        self, in_dim: int, mlp_dim: int, out_dim: int, unit_rows: bool = False
    ):
        super().__init__()
        if unit_rows:
            self.inp = UnitRowLinear(in_dim, mlp_dim * 2)
            self.out = UnitRowLinear(mlp_dim, out_dim, gain=True)
        else:
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
