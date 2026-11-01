"""Adversarial loss terms on raw critic logits; each returns a scalar."""

import torch
import torch.nn.functional as F

from latentdragon_common.registry import Registry

GEN_LOSS = Registry("generator loss")
DISC_LOSS = Registry("discriminator loss")


@GEN_LOSS.register("hinge")
def hinge_gen_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Negative mean fake logit."""
    return -torch.mean(logits_fake)


@GEN_LOSS.register("logistic")
def logistic_gen_loss(logits_fake: torch.Tensor) -> torch.Tensor:
    """Non-saturating logistic generator loss."""
    return F.softplus(-logits_fake).mean()


@DISC_LOSS.register("hinge")
def hinge_disc_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor):
    """Taming hinge discriminator loss."""
    loss_real = torch.mean(F.relu(1.0 - logits_real))
    loss_fake = torch.mean(F.relu(1.0 + logits_fake))
    return 0.5 * (loss_real + loss_fake)


@DISC_LOSS.register("vanilla")
def vanilla_disc_loss(logits_real: torch.Tensor, logits_fake: torch.Tensor):
    """Logistic discriminator loss, averaged over both sides."""
    loss_real = torch.mean(F.softplus(-logits_real))
    loss_fake = torch.mean(F.softplus(logits_fake))
    return 0.5 * (loss_real + loss_fake)
