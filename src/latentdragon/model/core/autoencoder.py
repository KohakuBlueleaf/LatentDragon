"""`Autoencoder`: domain, encoder, latent def, decoder and head, with no loss knowledge."""

from dataclasses import replace

import torch
import torch.nn as nn

from latentdragon.domain import Domain
from latentdragon.model.core.contracts import Encoded, LatentDef
from latentdragon.model.nn.reach import Reach, is_local


class Autoencoder(nn.Module):
    """The three arrows over named slots; each part stays separately callable.

    `encode(x) -> Encoded`, `decode(latent) -> output latent`, `head(output latent) -> x'`.
    """

    def __init__(
        self,
        domain: Domain,
        encoder: nn.Module,
        latent: LatentDef,
        decoder: nn.Module,
        head: nn.Module,
    ):
        super().__init__()
        self.domain = domain
        self.encoder = encoder
        self.latent = latent
        self.decoder = decoder
        self.head = head

    def encode(self, x: dict, offset: int = 0) -> Encoded:
        """Data slots to the latent; `offset` is the global index of the first sample."""
        return self.latent.serialize(self.encoder(x), offset)

    def decode(self, latent: dict) -> dict:
        """A live or packed latent to the output latent."""
        return self.decoder(self.latent.deserialize(latent))

    def forward(self, x: dict, offset: int = 0) -> tuple[dict, Encoded]:
        """Encode, decode and render; returns the output slots and the encoded record."""
        encoded = self.encode(x, offset)
        return self.head(self.decode(encoded.latent)), encoded

    def reach(self, start: int = 0) -> Reach:
        """Reach of the decoder stages from `start` plus the head, on that stage's grid.

        A decoder without a declared reach has no exact overlap.
        """
        if not hasattr(self.decoder, "reach"):
            return Reach(None)
        reach = self.decoder.reach(start, getattr(self.head, "reach", 0))
        return reach if is_local([self.head]) else replace(reach, spatial=None)

    def bind_kernels(self, device: torch.device) -> None:
        """Forward kernel selection to the latent def."""
        self.latent.bind_kernels(device)
