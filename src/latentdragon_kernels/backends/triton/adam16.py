"""Fused `adam16`: 16-bit state in memory, fp32 in registers, optional stochastic rounding."""

import torch
import triton
import triton.language as tl
from torch import Tensor

from latentdragon_kernels.backends.triton._common import MIN_SM, launch
from latentdragon_kernels.constraints import requires
from latentdragon_kernels.ops import ADAM16
from latentdragon_kernels.registry import register

BLOCK = 1024
WARPS = 4
# fp32 weight of the lowest bit each 16-bit dtype keeps; 0 turns stochastic rounding off
KEPT_BIT = {torch.bfloat16: 1 << 16, torch.float16: 1 << 13}


@triton.jit
def _round_stochastic(x, draw, kept_bit: tl.constexpr):
    """fp32 to the fp32 value of a 16-bit neighbour, up with its distance as probability.

    Both 16-bit dtypes keep the top bits of a normal fp32 value, so a uniform addend in the
    dropped bits carries into the kept ones with exactly that probability.
    """
    bits = x.to(tl.int32, bitcast=True)
    return ((bits + (draw & (kept_bit - 1))) & -kept_bit).to(tl.float32, bitcast=True)


@triton.jit
def _kernel(
    p_ptr,
    g_ptr,
    m_ptr,
    r_ptr,
    bad_ptr,
    n_elements,
    lr,
    beta1,
    beta2,
    eps,
    weight_decay,
    m_scale,
    r_scale,
    seed,
    KEPT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    """One step of a block of elements; a set flag at `bad_ptr` stores nothing."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = (offs < n_elements) & (tl.load(bad_ptr) == 0.0)

    p = tl.load(p_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(g_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    m = tl.load(m_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    r = tl.load(r_ptr + offs, mask=mask, other=0.0).to(tl.float32)

    # Moments in fp32; the second one is stored as its root
    m = beta1 * m + (1.0 - beta1) * g
    r_new = tl.sqrt(beta2 * (r * r) + (1.0 - beta2) * (g * g))

    # Debiased update, with weight decay decoupled from the gradient
    update = (m * m_scale) / (r_new * r_scale + eps)
    p = p - lr * (update + weight_decay * p)

    if KEPT:
        p_draw = tl.randint(seed, offs).to(tl.int32, bitcast=True)
        m_draw = tl.randint(seed + 1, offs).to(tl.int32, bitcast=True)
        p = _round_stochastic(p, p_draw, KEPT)
        m = _round_stochastic(m, m_draw, KEPT)

    tl.store(p_ptr + offs, p.to(p_ptr.dtype.element_ty), mask=mask)
    tl.store(m_ptr + offs, m.to(m_ptr.dtype.element_ty), mask=mask)
    tl.store(r_ptr + offs, r_new.to(r_ptr.dtype.element_ty), mask=mask)


def adam16(
    param: Tensor,
    grad: Tensor,
    exp_avg: Tensor,
    exp_rms: Tensor,
    bad: Tensor,
    step: int,
    lr: float,
    betas: tuple[float, float],
    eps: float,
    weight_decay: float,
    stochastic: bool = False,
    seed: int = 0,
) -> None:
    """One AdamW step computed in fp32 (see `ops`); nothing moves when `bad` is set."""
    beta1, beta2 = betas
    count = param.numel()
    launch(
        _kernel,
        (triton.cdiv(count, BLOCK),),
        param.device,
        param,
        grad,
        exp_avg,
        exp_rms,
        bad.reshape(1),
        count,
        lr,
        beta1,
        beta2,
        eps,
        weight_decay,
        1.0 / (1.0 - beta1**step),
        1.0 / (1.0 - beta2**step) ** 0.5,
        seed + step,
        KEPT=KEPT_BIT[param.dtype] if stochastic else 0,
        BLOCK=BLOCK,
        num_warps=WARPS,
    )


register(ADAM16, "triton", adam16, requires(min_sm=MIN_SM))
