"""Latent views: `(x, latent) -> (x', latent')`, each decoded and reconstructed on its own."""

from latentdragon_common.registry import Registry

LATENT_VIEW = Registry("latent view")


class LatentView:
    """A signal and latent transform; see docs/concepts/latent-slots.md for its flags."""

    adversarial = False
    codebook_safe = False
    domains: tuple[str, ...] = ("image", "video")
    weight = 1.0

    def __call__(self, x: dict, latent: dict, step: int) -> tuple[dict, dict]:
        """Return the transformed data slots and latent; subclasses implement it."""
        raise NotImplementedError
