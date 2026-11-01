"""`Autoencoder`: encoder, bottleneck, trunk and head, with no loss knowledge."""

import torch
import torch.nn as nn
from torch import Tensor

from latentdragon.domain import Domain
from latentdragon.types import BottleneckOut


class Autoencoder(nn.Module):
    """Each part is separately callable; `forward` composes them."""

    def __init__(
        self,
        encoder: nn.Module,
        bottleneck: nn.Module,
        trunk: nn.Module,
        head: nn.Module,
        domain: Domain | None = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.trunk = trunk
        self.head = head
        self.domain = domain

    def encode(self, x: Tensor) -> Tensor:
        """Raw encoder output, channels at dim 1."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Trunk features for a bottleneck latent."""
        return self.trunk(z)

    def bind_kernels(self, device: torch.device) -> None:
        """Forward kernel selection to the bottleneck."""
        self.bottleneck.bind_kernels(device)

    def forward(self, x: Tensor, offset: int = 0) -> tuple[Tensor, BottleneckOut]:
        """Encode, bottleneck, decode; returns the output and the bottleneck record."""
        out = self.bottleneck(self.encode(x), offset)
        return self.head(self.decode(out.z)), out
