"""Eager `adam16`: the kernel's update in torch ops, valid on every device."""

import torch
from torch import Tensor

from latentdragon_kernels.ops import ADAM16
from latentdragon_kernels.registry import register

# Low bits an fp32 value drops in each 16-bit dtype
DROPPED_BITS = {torch.bfloat16: 16, torch.float16: 13}


def round_stochastic(x: Tensor, dtype: torch.dtype, generator) -> Tensor:
    """fp32 `x` on the grid of `dtype`, away from zero with its distance as probability."""
    dropped = 1 << DROPPED_BITS[dtype]
    draw = torch.randint(
        0, dropped, x.shape, device=x.device, dtype=torch.int32, generator=generator
    )
    return ((x.view(torch.int32) + draw) & -dropped).view(torch.float32)


@torch.no_grad()
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
    if bad.item():
        return
    beta1, beta2 = betas
    g = grad.float()
    p = param.float()

    # Moments in fp32; the second one is stored as its root
    m = exp_avg.float().lerp_(g, 1.0 - beta1)
    rms = (
        exp_rms.float().square_().mul_(beta2).addcmul_(g, g, value=1.0 - beta2).sqrt_()
    )

    # Debiased update, with weight decay decoupled from the gradient
    denom = rms.mul(1.0 / (1.0 - beta2**step) ** 0.5).add_(eps)
    update = m.div(1.0 - beta1**step).div_(denom)
    p = p.sub_(update.add_(p, alpha=weight_decay), alpha=lr)

    # Writeback, rounded by distance when asked
    if stochastic:
        generator = torch.Generator(param.device).manual_seed(seed + step)
        p = round_stochastic(p, param.dtype, generator)
        m = round_stochastic(m, param.dtype, generator)
    param.copy_(p)
    exp_avg.copy_(m)
    exp_rms.copy_(rms)


register(ADAM16, "eager", adam16)
