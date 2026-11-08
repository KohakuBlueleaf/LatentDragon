"""Latent defs: the serdes between encoder slots and decoder slots."""

from latentdragon.model.latents.base import IdentityLatent, TensorLatent
from latentdragon.model.latents.bottleneck import Bottleneck
from latentdragon.model.latents.kl import KLLatent, kl_standard_normal
from latentdragon.model.latents.nested import NestedLatent
from latentdragon.model.latents.slots import SlotLatents
from latentdragon.model.latents.vq import VQLatent

__all__ = [
    # base classes
    "TensorLatent",
    # concrete classes
    "Bottleneck",
    "IdentityLatent",
    "KLLatent",
    "NestedLatent",
    "SlotLatents",
    "VQLatent",
    # functions
    "kl_standard_normal",
]
