"""MLP projection on `bmm` over hidden chunks: no hidden tensor outlives its chunk.

Autograd would hold the full `(rows, hidden)` activation until backward; here backward
recomputes one chunk at a time and writes each weight's gradient directly.
"""

import math

import torch
import torch.nn.functional as F
from torch import Tensor

from latentdragon_kernels.backends.eager.quantize import wide
from latentdragon_kernels.constraints import requires
from latentdragon_kernels.ops import MLP_PROJECT
from latentdragon_kernels.registry import register

CHUNK = 128  # hidden columns computed at once
_ACT = {"silu": F.silu, "gelu": F.gelu, "mish": F.mish}


def _silu_grad(x: Tensor) -> Tensor:
    sigmoid = x.sigmoid()
    return sigmoid * (1 + x * (1 - sigmoid))


def _gelu_grad(x: Tensor) -> Tensor:
    density = (-0.5 * x * x).exp() / math.sqrt(2 * math.pi)
    return 0.5 * (1 + torch.erf(x / math.sqrt(2))) + x * density


def _mish_grad(x: Tensor) -> Tensor:
    tanh = F.softplus(x).tanh()
    return tanh + x * (1 - tanh * tanh) * x.sigmoid()


_ACT_GRAD = {"silu": _silu_grad, "gelu": _gelu_grad, "mish": _mish_grad}


def _chunks(hidden: int) -> list[slice]:
    """The hidden columns of each chunk."""
    return [slice(s, min(s + CHUNK, hidden)) for s in range(0, hidden, CHUNK)]


class _MLPProjectFunction(torch.autograd.Function):
    """Chunked forward + chunked backward; saves the inputs, never the activations."""

    @staticmethod
    def forward(ctx, base, w1, b1, w2, b2, act, trainable_base):
        """Sum each chunk's contribution to `out (groups, rows, out_dim)`."""
        ctx.save_for_backward(base, w1, b1, w2, b2)
        ctx.act, ctx.trainable_base = act, trainable_base

        out = b2.expand(-1, base.shape[1], -1).clone()
        with torch.autocast(base.device.type, enabled=False):
            for part in _chunks(w1.shape[2]):
                pre = torch.bmm(base, w1[:, :, part]) + b1[:, :, part]
                out += torch.bmm(_ACT[act](pre), w2[:, part])
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Write the five gradients, one hidden chunk at a time."""
        base, w1, b1, w2, b2 = ctx.saved_tensors
        act, trainable_base = ctx.act, ctx.trainable_base
        grad_base = torch.zeros_like(base) if trainable_base else None
        grad_w1, grad_b1 = torch.empty_like(w1), torch.empty_like(b1)
        grad_w2 = torch.empty_like(w2)

        with torch.autocast(base.device.type, enabled=False):
            grad_out = grad_out.to(base.dtype)
            for part in _chunks(w1.shape[2]):
                w1_part, w2_part = w1[:, :, part], w2[:, part]

                # This chunk's activation and its gradient, at the output then the input
                pre = torch.bmm(base, w1_part) + b1[:, :, part]
                grad_act = torch.bmm(grad_out, w2_part.mT)
                grad_pre = grad_act * _ACT_GRAD[act](pre)

                grad_w2[:, part] = torch.bmm(_ACT[act](pre).mT, grad_out)
                grad_w1[:, :, part] = torch.bmm(base.mT, grad_pre)
                grad_b1[:, :, part] = grad_pre.sum(dim=1, keepdim=True)
                if trainable_base:
                    grad_base += torch.bmm(grad_pre, w1_part.mT)
        return (
            grad_base,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_out.sum(dim=1, keepdim=True),
            None,
            None,
        )


def mlp_project(
    base: Tensor,
    w1: Tensor,
    b1: Tensor,
    w2: Tensor,
    b2: Tensor,
    act: str = "silu",
    trainable_base: bool = True,
) -> Tensor:
    """MLP projection over hidden chunks (see `ops`)."""
    if act not in _ACT:
        raise ValueError(f"act must be one of {list(_ACT)}, got {act!r}")
    dtype = wide(torch.promote_types(base.dtype, w1.dtype))
    return _MLPProjectFunction.apply(
        base.to(dtype),
        w1.to(dtype),
        b1.to(dtype),
        w2.to(dtype),
        b2.to(dtype),
        act,
        trainable_base,
    )


# Only cards whose BLAS was measured against the Triton kernel
register(MLP_PROJECT, "gemm", mlp_project, requires(families=("blackwell-consumer",)))
