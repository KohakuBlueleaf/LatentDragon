"""Diagonal Gaussian posterior against a standard normal."""

import torch
from torch import Tensor, nn

from latentdragon.bottleneck.base import BOTTLENECK, Bottleneck
from latentdragon.types import BottleneckOut


def kl_standard_normal(mean: Tensor, logvar: Tensor) -> Tensor:
    """KL(N(mean, exp(logvar)) || N(0, I)), summed per sample, meaned over batch."""
    kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
    return kl.flatten(1).sum(dim=1).mean()


@BOTTLENECK.register("kl")
class KLBottleneck(Bottleneck):
    """Split the encoder output into mean and logvar, sample while training."""

    def __init__(
        self,
        latent_dim: int | None = None,
        logvar_clamp: tuple[float, float] = (-30.0, 20.0),
    ):
        super().__init__()
        self.logvar_clamp = logvar_clamp

    def in_channels(self, latent_dim: int) -> int:
        """Mean and logvar."""
        return 2 * latent_dim

    def init_encoder_out(self, layer: nn.Module, latent_dim: int) -> None:
        """Zero the logvar half of a layer whose outputs are at weight dim 0."""
        layer.weight.data[latent_dim:].zero_()
        layer.bias.data[latent_dim:].zero_()

    def forward(self, raw: Tensor, offset: int = 0) -> BottleneckOut:
        """Sample the posterior while training, else use the mean; report the KL."""
        # Split the channels into mean and clamped logvar
        mean, logvar = raw.chunk(2, dim=1)
        logvar = logvar.clamp(*self.logvar_clamp)

        # Reparameterized sample in training, mean at eval
        z = mean
        if self.training:
            z = mean + (0.5 * logvar).exp() * torch.randn_like(mean)

        kl = kl_standard_normal(mean, logvar)
        return BottleneckOut(z=z, aux={"kl": kl}, metrics={"kl": kl.detach()}, raw=mean)
