"""Regularizers on the slot-boundary states of the main pass."""

import torch

from latentdragon.latent.context import LatentCtx
from latentdragon.latent.terms import LATENT_TERM, LatentTerm
from latentdragon.model.core import tree

# The part whose parameters write each state
STATE_PARTS = {"encoder": "encoder", "latent": "encoder", "decoder": "decoder"}


@LATENT_TERM.register("state_rms")
class StateRms(LatentTerm):
    """Soft RMS target on a slot-boundary state: the mean of `(rms - target) ** 2`.

    `at` names the state: `encoder` (the encoder output, before the latent def), `latent`, or
    `decoder` (the output latent the head reads). `slot` picks one tensor by its slot path;
    `None` takes every float tensor of the state. The RMS runs over `channel_dim` split into
    `groups` equal groups, at every position, in fp32: the statistic an RMS norm with `groups`
    partitions divides by. The loss averages over positions, groups and tensors.
    """

    def __init__(
        self,
        at: str = "decoder",
        slot: str | None = None,
        target: float = 1.0,
        groups: int = 1,
        channel_dim: int = 1,
        eps: float = 1e-6,
    ):
        super().__init__()
        if at not in STATE_PARTS:
            raise ValueError(f"state {at!r} is not one of {sorted(STATE_PARTS)}")
        self.at = at
        self.slot = slot
        self.target = target
        self.groups = groups
        self.channel_dim = channel_dim
        self.eps = eps

    @classmethod
    def parts_of(cls, spec: dict) -> tuple[str, ...]:
        return (STATE_PARTS[spec.get("at", "decoder")],)

    def rms(self, value: torch.Tensor) -> torch.Tensor:
        """Per-position, per-group RMS of `value` in fp32, groups on the last dim."""
        channels = value.float().movedim(self.channel_dim, -1)
        grouped = channels.unflatten(-1, (self.groups, -1))
        return grouped.square().mean(-1).add(self.eps).sqrt()

    def forward(self, ctx: LatentCtx) -> tuple:
        found = tree.named_tensors(ctx.states[self.at])
        if self.slot is not None:
            found = {self.slot: found[self.slot]}
        found = {p: v for p, v in found.items() if v.is_floating_point()}
        if not found:
            raise ValueError(f"state {self.at!r} has no float tensor to regularize")

        # One loss per tensor; logs carry the loss and the mean RMS it pulls on
        losses, logs = [], {}
        for path, value in found.items():
            rms = self.rms(value)
            loss = (rms - self.target).square().mean()
            losses.append(loss)
            key = f"train/state_rms/{self.at}/{path}"
            logs[f"{key}_loss"] = loss.detach()
            logs[f"{key}_rms"] = rms.detach().mean()
        return torch.stack(losses).mean(), logs
