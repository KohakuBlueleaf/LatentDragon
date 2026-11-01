"""Schemes: importing this package registers every built-in."""

from latentdragon.gan.schemes.r3gan import r3gan
from latentdragon.gan.schemes.stylegan2 import stylegan2
from latentdragon.gan.schemes.taming import taming

__all__ = [
    # builders
    "r3gan",
    "stylegan2",
    "taming",
]
