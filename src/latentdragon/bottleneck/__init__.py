"""Latent bottlenecks: identity, kl, vq, nested."""

# Registry, base class and identity
from latentdragon.bottleneck.base import BOTTLENECK, Bottleneck, IdentityBottleneck

# Concrete bottlenecks
from latentdragon.bottleneck.kl import KLBottleneck
from latentdragon.bottleneck.nested import NestedBottleneck
from latentdragon.bottleneck.vq import VQBottleneck

__all__ = [
    # registries
    "BOTTLENECK",
    # base classes
    "Bottleneck",
    # concrete classes
    "IdentityBottleneck",
    "KLBottleneck",
    "NestedBottleneck",
    "VQBottleneck",
]
