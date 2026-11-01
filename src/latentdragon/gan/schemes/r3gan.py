"""R3GAN: relativistic pairing with R1 and R2 from one shared critic forward."""

import functools

import torch.nn.functional as F

from latentdragon.gan.scheme import (
    GAN_SCHEME,
    GanCtx,
    GanScheme,
    gen_backward,
    seed_of,
)
from latentdragon.gan.schemes.shared import grad_penalty, leaf_forward


def _active(disc_weight: float, gamma: float, ctx: GanCtx) -> dict:
    micro_count = ctx.micro_count
    real, fake, d_real, d_fake = leaf_forward(ctx)
    g_loss = F.softplus(-(d_fake - d_real.detach())).mean()
    g_grad = seed_of(g_loss, fake, ctx.seed_scale, retain_graph=True)
    gen_backward(ctx, (disc_weight / micro_count) * g_grad)

    d_adv = F.softplus(-(d_real - d_fake)).mean()
    r1 = grad_penalty(d_real, real, ctx.seed_scale)
    r2 = grad_penalty(d_fake, fake, ctx.seed_scale)
    d_loss = d_adv + 0.5 * gamma * (r1 + r2)
    ctx.backward_d(d_loss / micro_count)
    return {
        "g_loss": g_loss.detach(),
        "d_loss": d_loss.detach(),
        "r1": r1.detach(),
        "r2": r2.detach(),
        "logits_real": d_real.detach().mean(),
        "logits_fake": d_fake.detach().mean(),
    }


@GAN_SCHEME.register("r3gan")
def r3gan(
    disc_start: int = 0, disc_weight: float = 0.5, gamma: float = 1.0
) -> GanScheme:
    """Relativistic GAN with zero-centred R1 and R2 penalties."""
    return GanScheme(
        "r3gan", disc_start, functools.partial(_active, disc_weight, gamma)
    )
