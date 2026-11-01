"""Triton MLP projection: w2 @ act(w1 @ base + b1) + b2 without storing the hidden tensor.

Layout: base (groups, rows, in_dim), w1 (groups, in_dim, hidden), b1 (groups, 1, hidden),
w2 (groups, hidden, out_dim), b2 (groups, 1, out_dim).
"""

import torch
import triton
import triton.language as tl
from torch import Tensor

from latentdragon_kernels.backends.triton._common import (
    ERRORS,
    MIN_SM,
    launch,
    pad,
    tiles,
)
from latentdragon_kernels.constraints import requires
from latentdragon_kernels.ops import MLP_PROJECT
from latentdragon_kernels.registry import register
from latentdragon_kernels.tuning import tuner

_ACT_ID = {"silu": 0, "gelu": 1, "mish": 2}

# Candidate tile configs per kernel; backward uses smaller blocks
FORWARD = tiles("block_rows", "block_hidden")
BACKWARD = tiles("block_rows", "block_hidden", top=64)


@triton.jit
def _softplus(x):
    """Stable softplus: log(1 + exp(x))."""
    return tl.where(x > 20.0, x, tl.extra.cuda.libdevice.log1p(tl.exp(x)))


@triton.jit
def _act_forward(x, activation: tl.constexpr):
    """The activation of `x`."""
    if activation == 0:  # SiLU: x * sigmoid(x)
        return x * tl.sigmoid(x)
    elif activation == 1:  # GELU, tanh approximation
        cst = 0.7978845608028654  # sqrt(2/pi)
        inner = cst * (x + 0.044715 * x * x * x)
        return 0.5 * x * (1.0 + tl.extra.cuda.libdevice.tanh(inner))
    else:  # Mish: x * tanh(softplus(x))
        return x * tl.extra.cuda.libdevice.tanh(_softplus(x))


@triton.jit
def _act_backward(x, activation: tl.constexpr):
    """The activation derivative at `x`."""
    if activation == 0:  # SiLU': sigmoid(x) * (1 + x * (1 - sigmoid(x)))
        sig = tl.sigmoid(x)
        return sig * (1.0 + x * (1.0 - sig))
    elif activation == 1:  # GELU', tanh approximation
        cst = 0.7978845608028654
        inner = cst * (x + 0.044715 * x * x * x)
        tanh_val = tl.extra.cuda.libdevice.tanh(inner)
        dtanh = 1.0 - tanh_val * tanh_val
        return 0.5 * (1.0 + tanh_val) + 0.5 * x * dtanh * cst * (
            1.0 + 3.0 * 0.044715 * x * x
        )
    else:  # Mish': tanh(sp) + x * (1 - tanh(sp)^2) * sigmoid(x)
        tanh_sp = tl.extra.cuda.libdevice.tanh(_softplus(x))
        return tanh_sp + x * (1.0 - tanh_sp * tanh_sp) * tl.sigmoid(x)


@triton.jit
def _fwd_kernel(
    # Pointers
    base_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    b2_ptr,
    out_ptr,
    # Dimensions
    rows,
    in_dim: tl.constexpr,
    hidden,
    out_dim: tl.constexpr,
    # Padded dimensions for tl.dot
    in_pad: tl.constexpr,
    out_pad: tl.constexpr,
    # Strides for base: (groups, rows, in_dim)
    stride_base_group,
    stride_base_row,
    # Strides for w1: (groups, in_dim, hidden)
    stride_w1_group,
    stride_w1_in,
    # Strides for b1: (groups, 1, hidden)
    stride_b1_group,
    # Strides for w2: (groups, hidden, out_dim)
    stride_w2_group,
    stride_w2_hidden,
    # Strides for b2: (groups, 1, out_dim)
    stride_b2_group,
    # Strides for out: (groups, rows, out_dim)
    stride_out_group,
    stride_out_row,
    # 0 SiLU, 1 GELU, 2 Mish
    activation: tl.constexpr,
    # Block sizes
    block_rows: tl.constexpr,
    block_hidden: tl.constexpr,
):
    pid_group = tl.program_id(0)
    pid_row = tl.program_id(1)

    row_start = pid_row * block_rows
    offs_row = row_start + tl.arange(0, block_rows)
    row_mask = offs_row < rows

    offs_in = tl.arange(0, in_pad)
    in_mask = offs_in < in_dim
    offs_out = tl.arange(0, out_pad)
    out_mask = offs_out < out_dim

    # Load base tile: (block_rows, in_pad), zero-padded
    base_ptrs = (
        base_ptr
        + pid_group * stride_base_group
        + offs_row[:, None] * stride_base_row
        + offs_in[None, :]
    )
    base_tile = tl.load(
        base_ptrs, mask=(row_mask[:, None] & in_mask[None, :]), other=0.0
    ).to(tl.float32)

    # Load b2: (out_pad,)
    b2_ptrs = b2_ptr + pid_group * stride_b2_group + offs_out
    b2_val = tl.load(b2_ptrs, mask=out_mask, other=0.0).to(tl.float32)

    # Output accumulator: (block_rows, out_pad)
    acc = tl.zeros([block_rows, out_pad], dtype=tl.float32)
    acc += b2_val[None, :]

    # Loop over the hidden dim in block_hidden chunks
    for hidden_start in range(0, hidden, block_hidden):
        offs_hidden = hidden_start + tl.arange(0, block_hidden)
        hidden_mask = offs_hidden < hidden

        # Load w1 chunk: (in_pad, block_hidden)
        w1_ptrs = (
            w1_ptr
            + pid_group * stride_w1_group
            + offs_in[:, None] * stride_w1_in
            + offs_hidden[None, :]
        )
        w1_chunk = tl.load(
            w1_ptrs, mask=(in_mask[:, None] & hidden_mask[None, :]), other=0.0
        ).to(tl.float32)

        # Load b1 chunk: (block_hidden,)
        b1_ptrs = b1_ptr + pid_group * stride_b1_group + offs_hidden
        b1_chunk = tl.load(b1_ptrs, mask=hidden_mask, other=0.0).to(tl.float32)

        # First matmul: (block_rows, block_hidden)
        hidden_act = tl.dot(base_tile, w1_chunk, input_precision="ieee")
        hidden_act = _act_forward(hidden_act + b1_chunk[None, :], activation)

        # Load w2 chunk: (block_hidden, out_pad)
        w2_ptrs = (
            w2_ptr
            + pid_group * stride_w2_group
            + offs_hidden[:, None] * stride_w2_hidden
            + offs_out[None, :]
        )
        w2_chunk = tl.load(
            w2_ptrs, mask=(hidden_mask[:, None] & out_mask[None, :]), other=0.0
        ).to(tl.float32)

        # Second matmul: (block_rows, out_pad)
        acc += tl.dot(hidden_act, w2_chunk, input_precision="ieee")

    # Store output, without padding and out-of-bounds rows
    out_ptrs = (
        out_ptr
        + pid_group * stride_out_group
        + offs_row[:, None] * stride_out_row
        + offs_out[None, :]
    )
    tl.store(out_ptrs, acc, mask=(row_mask[:, None] & out_mask[None, :]))


@triton.jit
def _bwd_kernel(
    # Pointers, inputs
    base_ptr,
    w1_ptr,
    b1_ptr,
    w2_ptr,
    grad_out_ptr,
    # Pointers, outputs
    grad_base_ptr,
    grad_w1_ptr,
    grad_b1_ptr,
    grad_w2_ptr,
    grad_b2_ptr,
    # Dimensions
    rows,
    in_dim: tl.constexpr,
    hidden,
    out_dim: tl.constexpr,
    # Padded dimensions for tl.dot
    in_pad: tl.constexpr,
    out_pad: tl.constexpr,
    # Strides for base: (groups, rows, in_dim)
    stride_base_group,
    stride_base_row,
    # Strides for w1: (groups, in_dim, hidden)
    stride_w1_group,
    stride_w1_in,
    # Strides for b1: (groups, 1, hidden)
    stride_b1_group,
    # Strides for w2: (groups, hidden, out_dim)
    stride_w2_group,
    stride_w2_hidden,
    # Strides for grad_out: (groups, rows, out_dim)
    stride_go_group,
    stride_go_row,
    # Strides for grad_base: (groups, rows, in_dim)
    stride_gb_group,
    stride_gb_row,
    # Strides for grad_w1: (groups, in_dim, hidden)
    stride_gw1_group,
    stride_gw1_in,
    # Strides for grad_b1: (groups, 1, hidden)
    stride_gb1_group,
    # Strides for grad_w2: (groups, hidden, out_dim)
    stride_gw2_group,
    stride_gw2_hidden,
    # Strides for grad_b2: (groups, 1, out_dim)
    stride_gb2_group,
    # 0 SiLU, 1 GELU, 2 Mish
    activation: tl.constexpr,
    # Whether to compute grad_base
    trainable_base: tl.constexpr,
    # Block sizes
    block_rows: tl.constexpr,
    block_hidden: tl.constexpr,
):
    pid_group = tl.program_id(0)
    pid_row = tl.program_id(1)

    row_start = pid_row * block_rows
    offs_row = row_start + tl.arange(0, block_rows)
    row_mask = offs_row < rows

    offs_in = tl.arange(0, in_pad)
    in_mask = offs_in < in_dim
    offs_out = tl.arange(0, out_pad)
    out_mask = offs_out < out_dim

    # Load base tile: (block_rows, in_pad)
    base_ptrs = (
        base_ptr
        + pid_group * stride_base_group
        + offs_row[:, None] * stride_base_row
        + offs_in[None, :]
    )
    base_tile = tl.load(
        base_ptrs, mask=(row_mask[:, None] & in_mask[None, :]), other=0.0
    ).to(tl.float32)

    # Load grad_out tile: (block_rows, out_pad)
    go_ptrs = (
        grad_out_ptr
        + pid_group * stride_go_group
        + offs_row[:, None] * stride_go_row
        + offs_out[None, :]
    )
    grad_tile = tl.load(
        go_ptrs, mask=(row_mask[:, None] & out_mask[None, :]), other=0.0
    ).to(tl.float32)

    # grad_b2 partial: sum over the row tile, (out_pad,)
    grad_b2_partial = tl.sum(grad_tile, axis=0)

    # grad_base accumulator: (block_rows, in_pad)
    if trainable_base:
        grad_base_acc = tl.zeros([block_rows, in_pad], dtype=tl.float32)

    # Loop over the hidden dim in block_hidden chunks
    for hidden_start in range(0, hidden, block_hidden):
        offs_hidden = hidden_start + tl.arange(0, block_hidden)
        hidden_mask = offs_hidden < hidden

        # Load w1 chunk: (in_pad, block_hidden)
        w1_ptrs = (
            w1_ptr
            + pid_group * stride_w1_group
            + offs_in[:, None] * stride_w1_in
            + offs_hidden[None, :]
        )
        w1_chunk = tl.load(
            w1_ptrs, mask=(in_mask[:, None] & hidden_mask[None, :]), other=0.0
        ).to(tl.float32)

        # Load b1 chunk: (block_hidden,)
        b1_ptrs = b1_ptr + pid_group * stride_b1_group + offs_hidden
        b1_chunk = tl.load(b1_ptrs, mask=hidden_mask, other=0.0).to(tl.float32)

        # Recompute the hidden activation: (block_rows, block_hidden)
        h_pre = tl.dot(base_tile, w1_chunk, input_precision="ieee") + b1_chunk[None, :]
        h_act = _act_forward(h_pre, activation)
        act_grad = _act_backward(h_pre, activation)

        # Load w2 chunk: (block_hidden, out_pad)
        w2_ptrs = (
            w2_ptr
            + pid_group * stride_w2_group
            + offs_hidden[:, None] * stride_w2_hidden
            + offs_out[None, :]
        )
        w2_chunk = tl.load(
            w2_ptrs, mask=(hidden_mask[:, None] & out_mask[None, :]), other=0.0
        ).to(tl.float32)

        # grad_w2 chunk: h_act^T @ grad_tile, (block_hidden, out_pad)
        grad_w2_chunk = tl.dot(tl.trans(h_act), grad_tile, input_precision="ieee")
        gw2_ptrs = (
            grad_w2_ptr
            + pid_group * stride_gw2_group
            + offs_hidden[:, None] * stride_gw2_hidden
            + offs_out[None, :]
        )
        tl.atomic_add(
            gw2_ptrs, grad_w2_chunk, mask=(hidden_mask[:, None] & out_mask[None, :])
        )

        # Gradient at the activation output, then at its input: (block_rows, block_hidden)
        grad_h_act = tl.dot(grad_tile, tl.trans(w2_chunk), input_precision="ieee")
        grad_h_pre = grad_h_act * act_grad

        # grad_w1 chunk: base_tile^T @ grad_h_pre, (in_pad, block_hidden)
        grad_w1_chunk = tl.dot(tl.trans(base_tile), grad_h_pre, input_precision="ieee")
        gw1_ptrs = (
            grad_w1_ptr
            + pid_group * stride_gw1_group
            + offs_in[:, None] * stride_gw1_in
            + offs_hidden[None, :]
        )
        tl.atomic_add(
            gw1_ptrs, grad_w1_chunk, mask=(in_mask[:, None] & hidden_mask[None, :])
        )

        # grad_b1 chunk: sum over rows, (block_hidden,)
        grad_b1_chunk = tl.sum(grad_h_pre, axis=0)
        gb1_ptrs = grad_b1_ptr + pid_group * stride_gb1_group + offs_hidden
        tl.atomic_add(gb1_ptrs, grad_b1_chunk, mask=hidden_mask)

        # grad_base: grad_h_pre @ w1_chunk^T, (block_rows, in_pad)
        if trainable_base:
            grad_base_acc += tl.dot(
                grad_h_pre, tl.trans(w1_chunk), input_precision="ieee"
            )

    # Store grad_base: each program owns its tile
    if trainable_base:
        gb_ptrs = (
            grad_base_ptr
            + pid_group * stride_gb_group
            + offs_row[:, None] * stride_gb_row
            + offs_in[None, :]
        )
        tl.store(
            gb_ptrs,
            grad_base_acc.to(grad_base_ptr.dtype.element_ty),
            mask=(row_mask[:, None] & in_mask[None, :]),
        )

    gb2_ptrs = grad_b2_ptr + pid_group * stride_gb2_group + offs_out
    tl.atomic_add(gb2_ptrs, grad_b2_partial, mask=out_mask)


class _MLPProjectFunction(torch.autograd.Function):
    """Fused forward + fused backward; saves the inputs, never the hidden activations."""

    @staticmethod
    def forward(ctx, base, w1, b1, w2, b2, act_id, trainable_base):
        """Launch the fused forward kernel; returns `out (groups, rows, out_dim)` in fp32."""
        ctx.save_for_backward(base, w1, b1, w2, b2)
        ctx.act_id, ctx.trainable_base = act_id, trainable_base

        groups, rows, in_dim = base.shape
        hidden, out_dim = w1.shape[2], w2.shape[2]
        out = torch.empty(
            (groups, rows, out_dim), device=base.device, dtype=torch.float32
        )

        # Kernel arguments in signature order: tensors, sizes, padded sizes, strides
        args = (
            base,
            w1,
            b1,
            w2,
            b2,
            out,
            rows,
            in_dim,
            hidden,
            out_dim,
            pad(in_dim),
            pad(out_dim),
            base.stride(0),
            base.stride(1),
            w1.stride(0),
            w1.stride(1),
            b1.stride(0),
            w2.stride(0),
            w2.stride(1),
            b2.stride(0),
            out.stride(0),
            out.stride(1),
        )

        def run(config: dict):
            grid = (groups, triton.cdiv(rows, config["block_rows"]))
            return lambda: launch(
                _fwd_kernel, grid, base.device, *args, activation=act_id, **config
            )

        # Pick the tile config for this shape bucket, then run it once
        key = tuner.shape_key(rows=rows, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
        run(tuner.select("mlp_fwd", base.device, key, FORWARD, run, ERRORS))()
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """Launch the fused backward kernel; returns grads for the five tensors."""
        base, w1, b1, w2, b2 = ctx.saved_tensors
        act_id, trainable_base = ctx.act_id, ctx.trainable_base

        groups, rows, in_dim = base.shape
        hidden, out_dim = w1.shape[2], w2.shape[2]
        grad_out = grad_out.contiguous()

        # Weight grads are accumulated atomically in fp32; the base grad is skipped if frozen
        grad_base = torch.empty_like(base) if trainable_base else base.new_empty(1)
        grad_w1 = torch.empty_like(w1, dtype=torch.float32)
        grad_b1 = torch.empty_like(b1, dtype=torch.float32)
        grad_w2 = torch.empty_like(w2, dtype=torch.float32)
        grad_b2 = torch.empty_like(b2, dtype=torch.float32)

        # Kernel arguments in signature order: tensors, sizes, padded sizes, strides
        args = (
            base,
            w1,
            b1,
            w2,
            grad_out,
            grad_base,
            grad_w1,
            grad_b1,
            grad_w2,
            grad_b2,
            rows,
            in_dim,
            hidden,
            out_dim,
            pad(in_dim),
            pad(out_dim),
            base.stride(0),
            base.stride(1),
            w1.stride(0),
            w1.stride(1),
            b1.stride(0),
            w2.stride(0),
            w2.stride(1),
            grad_out.stride(0),
            grad_out.stride(1),
            grad_base.stride(0) if trainable_base else 0,
            grad_base.stride(1) if trainable_base else 0,
            grad_w1.stride(0),
            grad_w1.stride(1),
            grad_b1.stride(0),
            grad_w2.stride(0),
            grad_w2.stride(1),
            grad_b2.stride(0),
        )

        def run(config: dict):
            grid = (groups, triton.cdiv(rows, config["block_rows"]))

            def call() -> None:
                # Atomic accumulators must start at zero on every launch
                for grad in (grad_w1, grad_b1, grad_w2, grad_b2):
                    grad.zero_()
                launch(
                    _bwd_kernel,
                    grid,
                    base.device,
                    *args,
                    activation=act_id,
                    trainable_base=trainable_base,
                    **config,
                )

            return call

        key = tuner.shape_key(rows=rows, in_dim=in_dim, hidden=hidden, out_dim=out_dim)
        run(tuner.select("mlp_bwd", base.device, key, BACKWARD, run, ERRORS))()

        # Cast the fp32 accumulators back to the parameter dtypes
        return (
            grad_base if trainable_base else None,
            grad_w1.to(w1.dtype),
            grad_b1.to(b1.dtype),
            grad_w2.to(w2.dtype),
            grad_b2.to(b2.dtype),
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
    """MLP projection through the fused kernels (see `ops`)."""
    if act not in _ACT_ID:
        raise ValueError(f"act must be one of {list(_ACT_ID)}, got {act!r}")
    return _MLPProjectFunction.apply(
        base.contiguous(),
        w1.contiguous(),
        b1.contiguous(),
        w2.contiguous(),
        b2.contiguous(),
        _ACT_ID[act],
        trainable_base,
    )


register(MLP_PROJECT, "triton", mlp_project, requires(min_sm=MIN_SM))
