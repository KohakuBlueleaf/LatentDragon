"""Shared forward and R1 helpers for the single-forward logistic schemes."""

import torch

from latentdragon.gan.scheme import GanCtx


def leaf_forward(ctx: GanCtx) -> tuple:
    """One `D(real)` and one `D(fake)` on detached leaves."""
    fake = ctx.m.fake.detach().requires_grad_(True)
    real = ctx.m.real.detach().requires_grad_(True)
    return real, fake, ctx.critic(real), ctx.critic(fake)


def grad_penalty(
    logits: torch.Tensor, inputs: torch.Tensor, scale: torch.Tensor | float = 1.0
) -> torch.Tensor:
    """Mean squared input-gradient norm, in fp32, keeping the graph.

    `scale` may be the loss scale as a device tensor; it multiplies the seed, so no host read.
    """
    count = logits.numel()
    seed = torch.ones_like(logits) * (scale / count)
    grad = torch.autograd.grad(
        logits, inputs, grad_outputs=seed, retain_graph=True, create_graph=True
    )[0]
    return (grad.float() * (count / scale)).square().flatten(1).sum(dim=1).mean()
