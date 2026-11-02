"""Diagonal Gaussian posterior against a standard normal."""

import torch
from torch import Tensor, nn

from latentdragon.model.core.contracts import LATENT, Encoded
from latentdragon.model.latents.base import TensorLatent


def kl_standard_normal(mean: Tensor, logvar: Tensor) -> Tensor:
    """KL(N(mean, exp(logvar)) || N(0, I)), summed per sample, meaned over batch."""
    kl = 0.5 * (mean.pow(2) + logvar.exp() - 1.0 - logvar)
    return kl.flatten(1).sum(dim=1).mean()


@LATENT.register("kl")
class KLLatent(TensorLatent):
    """Split the encoder slot into mean and logvar along dim 1, sample while training.

    The encoder emits twice the latent channels; the latent is the sample, or the mean in
    eval mode, and the KL is the `kl` loss.
    """

    def __init__(
        self,
        input: str = "h",
        output: str = "z",
        logvar_clamp: tuple[float, float] = (-30.0, 20.0),
    ):
        super().__init__(input, output)
        self.logvar_clamp = logvar_clamp

    def init_from_encoder(self, layer: nn.Module) -> None:
        """Zero the logvar half of a layer whose outputs are at weight dim 0."""
        half = layer.weight.shape[0] // 2
        layer.weight.data[half:].zero_()
        layer.bias.data[half:].zero_()

    def encode(self, raw: Tensor, offset: int) -> Encoded:
        """Sample the posterior while training, else use the mean; report the KL."""
        # Split the channels into mean and clamped logvar
        mean, logvar = raw.chunk(2, dim=1)
        logvar = logvar.clamp(*self.logvar_clamp)

        # Reparameterized sample in training, mean at eval
        z = mean
        if self.training:
            z = mean + (0.5 * logvar).exp() * torch.randn_like(mean)

        kl = kl_standard_normal(mean, logvar)
        return Encoded(latent={self.output: z}, losses={"kl": kl}, extras={"raw": mean})
