"""Latent terms and their weighting, resolved once at build time."""

import torch

from latentdragon.latent.context import LatentCtx
from latentdragon_common.registry import Registry

LATENT_TERM = Registry("latent term")
WEIGHTING = Registry("term weighting")


class LatentTerm(torch.nn.Module):
    """`forward(ctx) -> (loss, logs)`; see docs/concepts/latent-slots.md for its flags."""

    requires: tuple[str, ...] = ("encoder",)
    teacher: str | None = None
    group = "model"

    def forward(self, ctx: LatentCtx) -> tuple:
        """Compute `(loss, logs)` from the context; subclasses implement it."""
        raise NotImplementedError


@WEIGHTING.register("fixed")
class FixedWeight:
    """A constant weight."""

    def __init__(self, weight: float = 1.0):
        self.weight = weight

    def __call__(self, loss: torch.Tensor, ctx: LatentCtx, anchor) -> float:
        return self.weight


@WEIGHTING.register("scheduled")
class ScheduledWeight:
    """Zero before `start`, then a linear ramp to `weight` over `ramp` steps."""

    def __init__(self, weight: float = 1.0, start: int = 0, ramp: int = 0):
        self.weight = weight
        self.start = start
        self.ramp = ramp

    def __call__(self, loss: torch.Tensor, ctx: LatentCtx, anchor) -> float:
        progress = (ctx.step - self.start + 1) / max(self.ramp, 1)
        return self.weight * min(max(progress, 0.0), 1.0)


@WEIGHTING.register("adaptive")
class AdaptiveWeight:
    """`weight` times the reconstruction / term gradient-norm ratio at `anchor`."""

    def __init__(self, weight: float = 1.0, clip: tuple = (0.0, 1e4)):
        self.weight = weight
        self.clip = clip

    def __call__(self, loss: torch.Tensor, ctx: LatentCtx, anchor) -> float:
        # Gradients of the term and of the reconstruction at the anchor
        term = torch.autograd.grad(loss, anchor, retain_graph=True)[0]
        recon = torch.autograd.grad(
            ctx.fake, anchor, grad_outputs=ctx.seed, retain_graph=True
        )[0]

        ratio = recon.float().norm() / (term.float().norm() + 1e-4)
        return self.weight * ratio.clamp(*self.clip).detach()
