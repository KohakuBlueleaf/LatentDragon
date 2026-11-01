"""Adversarial components: critics, loss terms and step schemes."""

# Critics
from latentdragon.gan.critics import CRITIC, Critic

# Scheme contract, registry and shared helpers
from latentdragon.gan.scheme import (
    GAN_SCHEME,
    GanCtx,
    GanMicro,
    GanScheme,
    composed_scalar,
    gen_backward,
    seed_of,
)

# Built-in scheme builders (importing registers them)
from latentdragon.gan.schemes import r3gan, stylegan2, taming

__all__ = [
    # registries
    "CRITIC",
    "GAN_SCHEME",
    # base classes
    "Critic",
    "GanCtx",
    "GanMicro",
    "GanScheme",
    # builders and functions
    "composed_scalar",
    "gen_backward",
    "r3gan",
    "seed_of",
    "stylegan2",
    "taming",
]
