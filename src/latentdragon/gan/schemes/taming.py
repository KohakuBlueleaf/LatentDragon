"""Taming/FVQ scheme: hinge loss and the batch-level adaptive weight."""

import dataclasses

import torch

from latentdragon.gan.losses import DISC_LOSS, GEN_LOSS
from latentdragon.gan.scheme import (
    GAN_SCHEME,
    GanCtx,
    GanScheme,
    composed_scalar,
    recon_only,
    seed_of,
)


class Taming:
    """The cached state and the two step policies of one taming run."""

    def __init__(self, disc_weight, gen_loss, disc_loss, adaptive, adaptive_clip):
        self.disc_weight = disc_weight
        self.gen_loss = gen_loss
        self.disc_loss = disc_loss
        self.adaptive = adaptive
        self.adaptive_clip = adaptive_clip
        self.cache: list = []
        self.nll_last = None
        self.gan_last = None
        self.fake_pass = self._graph_fake_pass
        self.critic_fake = self._fresh_fake_logits

    def tracking_idle(self, ctx: GanCtx) -> dict:
        """Gated-off step that still feeds a stat-tracking critic."""
        ctx.critic(ctx.m.fake.contiguous())
        self._hold(ctx, None, None, None)
        if ctx.micro_index < ctx.micro_count - 1:
            return {}
        cached, self.cache = self.cache, []
        for index, (m, _, _, _) in enumerate(cached):
            ctx.backward_g(
                composed_scalar(m, m.recon_seed / ctx.micro_count, ctx.micro_count),
                final=index == len(cached) - 1,
            )
        for m, _, _, _ in cached:
            ctx.critic(m.real.contiguous().detach())
            ctx.critic(m.fake.contiguous().detach())
        return {}

    def active(self, ctx: GanCtx) -> dict:
        """Cache this micro-batch; finalize the whole batch on the last one."""
        micro_count, last_w = ctx.micro_count, ctx.anchor

        # Reset the anchor accumulators at the start of a step
        if ctx.micro_index == 0 and self.adaptive:
            self.nll_last = torch.zeros_like(last_w)
            self.gan_last = torch.zeros_like(last_w)

        # Cache this micro; wait for the last one
        g_loss, gan_seed, logits = self.fake_pass(ctx)
        if self.adaptive:
            self._accumulate_anchor(ctx, gan_seed)
        self._hold(ctx, gan_seed.detach(), g_loss.detach(), logits)
        if ctx.micro_index < micro_count - 1:
            return {}

        # Last micro: generator backward with the adaptive GAN weight, then critic
        d_adaptive = self._adaptive_ratio(ctx)
        scale = d_adaptive * self.disc_weight
        cached, self.cache = self.cache, []
        for index, (mm, seed, _, _) in enumerate(cached):
            ctx.backward_g(
                composed_scalar(
                    mm,
                    mm.recon_seed / micro_count + (scale / micro_count) * seed,
                    micro_count,
                ),
                final=index == len(cached) - 1,
            )
        return self._critic_pass(ctx, cached, d_adaptive, scale)

    def share_fake_forward(self) -> None:
        """Reuse one fake forward for both losses; valid without batch statistics."""
        self.fake_pass = self._leaf_fake_pass
        self.critic_fake = self._cached_fake_logits

    def _graph_fake_pass(self, ctx: GanCtx) -> tuple:
        """Critic forward on the live fake; returns `(g_loss, seed, None)`."""
        g_loss = self.gen_loss(ctx.critic(ctx.m.fake.contiguous()))
        return g_loss, seed_of(g_loss, ctx.m.fake, ctx.seed_scale), None

    def _leaf_fake_pass(self, ctx: GanCtx) -> tuple:
        """Critic forward on a detached fake leaf; returns `(g_loss, seed, logits)`."""
        fake = ctx.m.fake.detach().requires_grad_(True)
        logits = ctx.critic(fake)
        g_loss = self.gen_loss(logits)
        seed = seed_of(g_loss, fake, ctx.seed_scale, retain_graph=True)
        return g_loss, seed, logits

    @staticmethod
    def _fresh_fake_logits(ctx: GanCtx, micro, logits) -> torch.Tensor:
        return ctx.critic(micro.fake.contiguous().detach())

    @staticmethod
    def _cached_fake_logits(ctx: GanCtx, micro, logits) -> torch.Tensor:
        return logits

    def _hold(self, ctx: GanCtx, seed, g_loss, logits) -> None:
        """Keep this micro-batch until the last one of the step arrives."""
        self.cache.append((ctx.m, seed, g_loss, logits))

    def _accumulate_anchor(self, ctx: GanCtx, gan_seed: torch.Tensor) -> None:
        """Add this micro's nll and gan gradients at the head anchor weight."""
        m, micro_count, last_w = ctx.m, ctx.micro_count, ctx.anchor
        scale = ctx.seed_scale
        nll = torch.autograd.grad(
            m.fake, last_w, grad_outputs=m.recon_seed * scale, retain_graph=True
        )[0]
        gan = torch.autograd.grad(
            m.fake, last_w, grad_outputs=gan_seed * scale, retain_graph=True
        )[0]
        self.nll_last = self.nll_last + nll.detach() / (micro_count * scale)
        self.gan_last = self.gan_last + gan.detach() / (micro_count * scale)

    def _adaptive_ratio(self, ctx: GanCtx) -> torch.Tensor:
        """Clipped ratio of nll to gan gradient norms across ranks (1 if off)."""
        if not self.adaptive:
            return torch.ones((), device=ctx.m.fake.device)

        # One all-reduce for both anchor gradients
        both = ctx.sync.all_mean_(torch.stack([self.nll_last, self.gan_last]))
        low, high = self.adaptive_clip
        ratio = both[0].float().norm() / (both[1].float().norm() + 1e-4)
        return ratio.clamp(low, high).detach()

    def _critic_pass(self, ctx, cached, d_adaptive, scale) -> dict:
        """Critic backward over every cached micro; returns the step's GAN logs."""
        micro_count = ctx.micro_count
        d_total = torch.zeros((), device=ctx.m.fake.device)
        real_mean, fake_mean = 0.0, 0.0
        for index, (mm, _, _, held) in enumerate(cached):
            logits_real = ctx.critic(mm.real.contiguous().detach())
            logits_fake = self.critic_fake(ctx, mm, held)
            real_mean = real_mean + logits_real.detach().mean() / micro_count
            fake_mean = fake_mean + logits_fake.detach().mean() / micro_count
            d_loss = self.disc_weight * self.disc_loss(logits_real, logits_fake)
            ctx.backward_d(d_loss / micro_count, final=index == len(cached) - 1)
            d_total = d_total + d_loss.detach() / micro_count
        g_losses = torch.stack([g for _, _, g, _ in cached])
        return {
            "logits_real": real_mean,
            "logits_fake": fake_mean,
            "gan_scale": scale,
            "d_adaptive": d_adaptive,
            "g_loss": g_losses.mean(),
            "d_loss": d_total,
        }


@dataclasses.dataclass(frozen=True)
class TamingScheme(GanScheme):
    """Taming policy; idle critic forwards only exist for stat-tracking critics."""

    policy: Taming | None = None

    def for_critic(self, critic) -> GanScheme:
        if critic.tracks_stats:
            return dataclasses.replace(self, idle=self.policy.tracking_idle)
        self.policy.share_fake_forward()
        return self


@GAN_SCHEME.register("taming")
def taming(
    disc_start: int = 0,
    disc_weight: float = 0.5,
    gen_adv_loss: str = "hinge",
    disc_loss: str = "hinge",
    adaptive_weight: bool = True,
    adaptive_clip: tuple[float, float] = (0.1, 1e4),
) -> GanScheme:
    """Hinge GAN whose weight is rescaled at the head anchor."""
    policy = Taming(
        disc_weight,
        GEN_LOSS.get(gen_adv_loss),
        DISC_LOSS.get(disc_loss),
        adaptive_weight,
        adaptive_clip,
    )
    return TamingScheme(
        "taming",
        disc_start,
        policy.active,
        recon_only,
        retained_backward=True,
        policy=policy,
    )
