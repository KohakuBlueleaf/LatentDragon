"""Commitment, codebook and inverse VQ losses."""

from collections.abc import Callable

from torch import Tensor


def commitment_codebook(z_q: Tensor, z_p: Tensor) -> tuple[Tensor, Tensor]:
    """Mean squared distance with the codebook side, then the data side, detached."""
    commitment = ((z_q.detach() - z_p) ** 2).mean()
    codebook = ((z_q - z_p.detach()) ** 2).mean()
    return commitment, codebook


def inverse_vq_loss(codebook: Tensor, data: Tensor, nearest: Callable) -> Tensor:
    """Pull every code of `(P, N, D)` toward its nearest detached `(P, M, D)` point."""
    codes = codebook.detach().permute(1, 0, 2).contiguous()
    nearest_z, _, _ = nearest(codes, data.detach())
    nearest_z = nearest_z.permute(1, 0, 2).contiguous()
    return ((codebook - nearest_z) ** 2).mean()


def ivq_terms(
    codebook: Tensor, z_p: Tensor, parts: list[Tensor], nearest: Callable
) -> dict[str, Tensor]:
    """Inverse VQ per level of `(R, P, N, D)` on its residual, meaned over levels."""
    partitions, dim = z_p.shape[-2:]
    terms = []
    residual = z_p
    for level, z_q in zip(codebook.unbind(0), parts, strict=True):
        data = residual.reshape(-1, partitions, dim).permute(1, 0, 2).contiguous()
        terms.append(inverse_vq_loss(level, data, nearest))
        residual = residual - z_q
    return {"ivq_loss": sum(terms) / len(terms)}


def no_ivq_terms(
    codebook: Tensor, z_p: Tensor, parts: list[Tensor], nearest: Callable
) -> dict[str, Tensor]:
    """No inverse VQ term."""
    return {}
