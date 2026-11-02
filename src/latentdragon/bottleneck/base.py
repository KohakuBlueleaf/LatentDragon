"""`BOTTLENECK` registry, the bottleneck base and the identity bottleneck."""

import torch
from torch import Tensor, nn

from latentdragon.types import BottleneckOut
from latentdragon_common.registry import Registry

BOTTLENECK = Registry("bottleneck")


class Bottleneck(nn.Module):
    """Latent gate between encoder and trunk; channels at dim 1 for any rank."""

    def in_channels(self, latent_dim: int) -> int:
        """Channels the encoder must emit."""
        return latent_dim

    def init_encoder_out(self, layer: nn.Module, latent_dim: int) -> None:
        """Adjust the encoder output layer; nothing by default."""

    def eval_masks(self) -> list[int]:
        """Channel counts to evaluate separately; none by default."""
        return []

    def bind_kernels(self, device: torch.device | str) -> None:
        """Resolve kernels for `device`; nothing by default."""

    def track_stats(self, on: bool) -> None:
        """Turn statistics that only feed logs on or off; nothing by default."""


@BOTTLENECK.register("identity")
class IdentityBottleneck(Bottleneck):
    """The encoder output, untouched."""

    def __init__(self, latent_dim: int | None = None):
        super().__init__()

    def forward(self, raw: Tensor, offset: int = 0) -> BottleneckOut:
        """Pass `raw` through as the latent."""
        return BottleneckOut(z=raw, aux={}, metrics={}, raw=raw)
