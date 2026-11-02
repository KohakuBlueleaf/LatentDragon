"""Remote-code model for exported folders: a diffusers VAE API over an Autoencoder."""

from dataclasses import dataclass

import torch

try:
    from diffusers import ConfigMixin, ModelMixin
    from diffusers.configuration_utils import register_to_config
except ImportError:
    ConfigMixin = object
    ModelMixin = torch.nn.Module

    def register_to_config(fn):
        return fn


from latentdragon.io.arch_json import upgrade_arch
from latentdragon.model import build_autoencoder


@dataclass
class _Posterior:
    mean: torch.Tensor


@dataclass
class _Encoded:
    latent_dist: _Posterior


@dataclass
class _Decoded:
    sample: torch.Tensor


class LatentDragonVAE(ModelMixin, ConfigMixin):
    """`encode(x).latent_dist.mean` and `decode(z).sample`, as diffusers callers expect."""

    # records every __init__ argument in the config
    @register_to_config
    def __init__(
        self,
        arch: dict,
        latent_channels: int | None = None,
        block_out_channels: list[int] | None = None,
        latents_mean: list[float] | None = None,
        latents_std: list[float] | None = None,
    ):
        """Build the autoencoder from `arch`.

        Args:
            arch: the `ARCH` dict (the `arch` entry of the exported `config.json`).
            latent_channels, block_out_channels: the diffusers VAE config fields; the
                config gets the values derived from `arch` when they are None.
            latents_mean, latents_std: per-channel statistics for latent scaling,
                recorded in the config; 0 and 1 when None. The model does not apply them.
        """
        super().__init__()

        arch = upgrade_arch(arch)
        self.model = build_autoencoder(arch)

        # fill the config fields left as None from the arch
        channels = arch["decoder"]["in_channels"]
        scale = getattr(self.model.encoder, "spatial_scale", 1)
        self.register_to_config(
            latent_channels=latent_channels or channels,
            block_out_channels=block_out_channels or [channels] * scale.bit_length(),
            latents_mean=latents_mean or [0.0] * channels,
            latents_std=latents_std or [1.0] * channels,
        )

    def _slot(self, slots: dict) -> torch.Tensor:
        """The one tensor a diffusers caller exchanges."""
        if len(slots) != 1:
            raise ValueError(
                f"diffusers callers take one tensor; this model has {list(slots)}"
            )
        return next(iter(slots.values()))

    def encode(self, x: torch.Tensor) -> _Encoded:
        """Encode to the continuous latent view (the mean of a kl, the input of a vq)."""
        encoded = self.model.encode({self.model.domain.slots[0]: x})
        return _Encoded(_Posterior(self._slot(self.model.latent.raw(encoded))))

    def decode(self, z: torch.Tensor) -> _Decoded:
        """Decode a decoder-input latent to the output signal."""
        slots = {self.model.decoder.inputs[0]: z}
        return _Decoded(self._slot(self.model.head(self.model.decoder(slots))))
