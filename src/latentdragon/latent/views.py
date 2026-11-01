"""Latent views: `(x, z) -> (x', z')`, each decoded and reconstructed on its own."""

import torch

from latentdragon_common.registry import Registry

LATENT_VIEW = Registry("latent view")


class LatentView:
    """A signal and latent transform; see docs/concepts/latent-slots.md for its flags."""

    adversarial = False
    codebook_safe = False
    domains: tuple[str, ...] = ("image", "video")
    weight = 1.0

    def __call__(self, x: torch.Tensor, z: torch.Tensor, step: int) -> tuple:
        """Return the transformed `(x, z)`; subclasses implement it."""
        raise NotImplementedError
