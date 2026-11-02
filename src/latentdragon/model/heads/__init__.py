"""Heads: the output latent to data slots."""

from latentdragon.model.heads.conv import ConvHead
from latentdragon.model.heads.identity import IdentityHead

__all__ = ["ConvHead", "IdentityHead"]
