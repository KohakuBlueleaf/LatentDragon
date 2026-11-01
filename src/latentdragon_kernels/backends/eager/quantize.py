"""Eager quantize: full distance matrix per partition, scatter backward."""

from collections.abc import Callable

import torch
from torch import Tensor

from latentdragon_kernels.ops import NEAREST, QUANTIZE
from latentdragon_kernels.registry import register


def wide(dtype: torch.dtype) -> torch.dtype:
    """`dtype`, at least fp32."""
    return torch.promote_types(dtype, torch.float32)


@torch.no_grad()
def nearest(z: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Nearest code per partition: gathered codes, indices, squared distances."""
    shape = z.shape[:-2]
    parts, dim = z.shape[-2:]
    flat = z.reshape(-1, parts, dim)
    codebook = codebook.to(z.dtype)
    dtype = wide(z.dtype)

    out_zq = torch.empty_like(flat)
    out_indices = torch.empty(flat.shape[:2], device=z.device, dtype=torch.int64)
    out_dists = torch.empty(flat.shape[:2], device=z.device, dtype=dtype)
    with torch.autocast(z.device.type, enabled=False):
        for part in range(parts):
            z_part = flat[:, part].to(dtype)
            cb_part = codebook[part].to(dtype)

            # Squared L2: ||z||^2 + ||cb||^2 - 2 z.cb
            dists = (
                z_part.pow(2).sum(dim=1, keepdim=True)
                + cb_part.pow(2).sum(dim=1, keepdim=True).T
                - 2 * z_part @ cb_part.T
            )

            min_dists, min_indices = dists.min(dim=1)
            out_indices[:, part] = min_indices
            out_dists[:, part] = min_dists
            out_zq[:, part] = codebook[part][min_indices]
    return (
        out_zq.reshape(*shape, parts, dim),
        out_indices.reshape(*shape, parts),
        out_dists.reshape(*shape, parts),
    )


def scatter_grad(
    grad_zq: Tensor,
    grad_zq_diff: Tensor,
    indices_pm: Tensor,
    shape: torch.Size,
) -> Tensor:
    """Codebook gradient `(parts, codes, dim)` from both outputs, in at least fp32."""
    parts, codes, dim = shape
    dtype = wide(grad_zq.dtype)

    # Combined per-row gradient, partition-major
    grad = grad_zq.to(dtype) - grad_zq_diff.to(dtype)
    grad_pm = grad.reshape(-1, parts, dim).transpose(0, 1)

    # Sum the rows that selected the same code
    grad_codebook = grad_pm.new_zeros((parts, codes, dim))
    index = indices_pm.unsqueeze(-1).expand(-1, -1, dim)
    return grad_codebook.scatter_add_(1, index, grad_pm)


class QuantizeFunction(torch.autograd.Function):
    """`search(z, codebook)` forward + scatter backward, shared by every backend.

    grad_z = grad_zq_diff; grad_codebook = scatter(grad_zq - grad_zq_diff, indices).
    """

    @staticmethod
    def forward(ctx, z: Tensor, codebook: Tensor, search: Callable):
        """Search, then return `(z_q, z - z_q, indices, min_dists)`."""
        z_q, indices, min_dists = search(z, codebook)
        z_q_diff = z - z_q

        # Save partition-major indices for the scatter
        parts = z.shape[-2]
        ctx.save_for_backward(indices.reshape(-1, parts).transpose(0, 1).contiguous())
        ctx.shape, ctx.dtype = codebook.shape, codebook.dtype
        return z_q, z_q_diff, indices, min_dists

    @staticmethod
    def backward(ctx, grad_zq, grad_zq_diff, _grad_indices, _grad_dists):
        """Identity gradient to `z` from `z_q_diff`; scatter onto the codebook."""
        (indices_pm,) = ctx.saved_tensors
        grad_codebook = scatter_grad(grad_zq, grad_zq_diff, indices_pm, ctx.shape)
        return grad_zq_diff, grad_codebook.to(ctx.dtype), None


def quantize(z: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Nearest code per partition with scatter backward (see `ops`)."""
    return QuantizeFunction.apply(z, codebook, nearest)


register(QUANTIZE, "eager", quantize)
register(NEAREST, "eager", nearest)
