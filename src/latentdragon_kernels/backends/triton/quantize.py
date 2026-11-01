"""Triton quantize: fused distance + gather per partition, eager scatter backward."""

import torch
import triton
import triton.language as tl
from torch import Tensor

from latentdragon_kernels.backends.eager.quantize import QuantizeFunction
from latentdragon_kernels.backends.triton._common import ERRORS, MIN_SM, launch, tiles
from latentdragon_kernels.constraints import requires
from latentdragon_kernels.ops import NEAREST, QUANTIZE
from latentdragon_kernels.registry import register
from latentdragon_kernels.tuning import tuner

CANDIDATES = tiles("block_rows", "block_codes", "block_dim")


@triton.jit
def _kernel(
    # Pointers
    z_ptr,
    z_norm_ptr,
    cb_ptr,
    cb_norm_ptr,
    out_zq_ptr,
    out_indices_ptr,
    out_dists_ptr,
    # Dimensions
    rows,
    codes,
    dim,
    # Strides for z: (parts, rows, dim)
    stride_z_part,
    stride_z_row,
    # Strides for z_norm: (parts, rows)
    stride_zn_part,
    # Strides for cb: (parts, codes, dim)
    stride_cb_part,
    stride_cb_code,
    # Strides for cb_norm: (parts, codes)
    stride_cbn_part,
    # Strides for out_zq: (parts, rows, dim)
    stride_ozq_part,
    stride_ozq_row,
    # Strides for out_indices / out_dists: (parts, rows)
    stride_oi_part,
    stride_od_part,
    # Block sizes
    block_rows: tl.constexpr,
    block_codes: tl.constexpr,
    block_dim: tl.constexpr,
):
    # Grid: (parts, ceil(rows / block_rows))
    pid_part = tl.program_id(0)
    pid_row = tl.program_id(1)

    row_start = pid_row * block_rows
    offs_row = row_start + tl.arange(0, block_rows)
    row_mask = offs_row < rows

    z_base = z_ptr + pid_part * stride_z_part
    cb_base = cb_ptr + pid_part * stride_cb_part

    # Load z_norm tile: (block_rows,)
    zn_ptrs = z_norm_ptr + pid_part * stride_zn_part + offs_row
    z_norm_tile = tl.load(zn_ptrs, mask=row_mask, other=0.0)

    min_dists = tl.full([block_rows], float("inf"), dtype=tl.float32)
    min_indices = tl.zeros([block_rows], dtype=tl.int64)

    # Loop over the codebook in chunks of block_codes
    for code_start in range(0, codes, block_codes):
        offs_code = code_start + tl.arange(0, block_codes)
        code_mask = offs_code < codes

        cbn_ptrs = cb_norm_ptr + pid_part * stride_cbn_part + offs_code
        cb_norm_chunk = tl.load(cbn_ptrs, mask=code_mask, other=0.0)

        # Dot product chunked over dim, accumulated in fp32: (block_rows, block_codes)
        dots = tl.zeros([block_rows, block_codes], dtype=tl.float32)
        for dim_start in range(0, dim, block_dim):
            offs_dim = dim_start + tl.arange(0, block_dim)
            dim_mask = offs_dim < dim

            z_chunk = tl.load(
                z_base + offs_row[:, None] * stride_z_row + offs_dim[None, :],
                mask=row_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            cb_chunk = tl.load(
                cb_base + offs_code[:, None] * stride_cb_code + offs_dim[None, :],
                mask=code_mask[:, None] & dim_mask[None, :],
                other=0.0,
            )
            dots += tl.dot(z_chunk, tl.trans(cb_chunk))

        # Distances: ||z||^2 + ||cb||^2 - 2 * z @ cb^T
        dists = z_norm_tile[:, None] + cb_norm_chunk[None, :] - 2.0 * dots
        dists = tl.where(code_mask[None, :], dists, float("inf"))

        # Update the running min
        chunk_min_dists = tl.min(dists, axis=1)
        chunk_min_idx = tl.argmin(dists, axis=1)
        improved = chunk_min_dists < min_dists
        min_dists = tl.where(improved, chunk_min_dists, min_dists)
        min_indices = tl.where(
            improved,
            (chunk_min_idx + code_start).to(tl.int64),
            min_indices,
        )

    # Gather z_q: codebook[pid_part, min_indices[row], :]
    ozq_base = out_zq_ptr + pid_part * stride_ozq_part
    for dim_start in range(0, dim, block_dim):
        offs_dim = dim_start + tl.arange(0, block_dim)
        dim_mask = offs_dim < dim

        cb_gather = tl.load(
            cb_base + min_indices[:, None] * stride_cb_code + offs_dim[None, :],
            mask=row_mask[:, None] & dim_mask[None, :],
            other=0.0,
        )
        tl.store(
            ozq_base + offs_row[:, None] * stride_ozq_row + offs_dim[None, :],
            cb_gather,
            mask=row_mask[:, None] & dim_mask[None, :],
        )

    # Store indices and dists: (block_rows,)
    oi_ptrs = out_indices_ptr + pid_part * stride_oi_part + offs_row
    od_ptrs = out_dists_ptr + pid_part * stride_od_part + offs_row
    tl.store(oi_ptrs, min_indices, mask=row_mask)
    tl.store(od_ptrs, min_dists, mask=row_mask)


@torch.no_grad()
def nearest(z: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Nearest code per partition: gathered codes, indices, squared distances."""
    shape = z.shape[:-2]
    parts, dim = z.shape[-2:]
    codes = codebook.shape[1]

    # Partition-major layout and squared norms for the distance expansion
    z_pm = z.reshape(-1, parts, dim).transpose(0, 1).contiguous()  # (parts, rows, dim)
    rows = z_pm.shape[1]
    codebook = codebook.to(z.dtype).contiguous()

    z_norm = z_pm.float().pow(2).sum(dim=-1).contiguous()  # (parts, rows)
    cb_norm = codebook.float().pow(2).sum(dim=-1).contiguous()  # (parts, codes)

    out_zq = torch.empty_like(z_pm)
    out_indices = torch.empty((parts, rows), device=z.device, dtype=torch.int64)
    out_dists = torch.empty((parts, rows), device=z.device, dtype=torch.float32)

    # Kernel arguments in signature order: tensors, sizes, strides
    args = (
        z_pm,
        z_norm,
        codebook,
        cb_norm,
        out_zq,
        out_indices,
        out_dists,
        rows,
        codes,
        dim,
        z_pm.stride(0),
        z_pm.stride(1),
        z_norm.stride(0),
        codebook.stride(0),
        codebook.stride(1),
        cb_norm.stride(0),
        out_zq.stride(0),
        out_zq.stride(1),
        out_indices.stride(0),
        out_dists.stride(0),
    )

    def run(config: dict):
        grid = (parts, triton.cdiv(rows, config["block_rows"]))
        return lambda: launch(_kernel, grid, z.device, *args, **config)

    # Pick the tile config for this shape bucket, then run it once
    key = tuner.shape_key(rows=rows, codes=codes, dim=dim)
    run(tuner.select("quantize", z.device, key, CANDIDATES, run, ERRORS))()

    # (parts, rows, ...) -> (rows, parts, ...) -> (..., parts, ...)
    return (
        out_zq.transpose(0, 1).reshape(*shape, parts, dim),
        out_indices.transpose(0, 1).reshape(*shape, parts),
        out_dists.transpose(0, 1).reshape(*shape, parts),
    )


def quantize(z: Tensor, codebook: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Nearest code per partition with scatter backward (see `ops`)."""
    return QuantizeFunction.apply(z, codebook, nearest)


register(QUANTIZE, "triton", quantize, requires(min_sm=MIN_SM))
register(NEAREST, "triton", nearest, requires(min_sm=MIN_SM))
