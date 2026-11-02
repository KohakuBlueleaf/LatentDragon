"""Shared forward and R1 helpers for the single-forward logistic schemes."""

import torch

from latentdragon.gan.scheme import GanCtx, leaf
from latentdragon.model.core import tree


def leaf_forward(ctx: GanCtx) -> tuple:
    """One `D(real)` and one `D(fake)` on detached leaf slot dicts."""
    fake = leaf(ctx.m.fake)
    real = leaf(ctx.m.real)
    return real, fake, ctx.critic(real), ctx.critic(fake)


def grad_penalty(
    logits: torch.Tensor, inputs, scale: torch.Tensor | float = 1.0
) -> torch.Tensor:
    """Mean squared input-gradient norm over every slot, in fp32, keeping the graph.

    `scale` may be the loss scale as a device tensor; it multiplies the seed, so no host read.
    """
    count = logits.numel()
    seed = torch.ones_like(logits) * (scale / count)
    grads = tree.grad(
        logits, inputs, grad_outputs=seed, retain_graph=True, create_graph=True
    )
    squares = [
        (grad.float() * (count / scale)).square().flatten(1).sum(dim=1)
        for grad in tree.leaves(grads)
    ]
    return sum(squares).mean()
