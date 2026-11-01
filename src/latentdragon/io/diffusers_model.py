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


from latentdragon.io.arch_json import arch_from_json, diffusers_levels
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
            arch: the `arch_to_json` form of an `AEArchConfig` (the `arch` entry of
                the exported `config.json`).
            latent_channels, block_out_channels: the diffusers VAE config fields; the
                config gets the values derived from `arch` when they are None.
            latents_mean, latents_std: per-channel statistics for latent scaling,
                recorded in the config; 0 and 1 when None. The model does not apply them.
        """
        super().__init__()

        config = arch_from_json(arch)
        self.model = build_autoencoder(config)

        # fill the config fields left as None from the arch
        channels = config.latent_dim
        self.register_to_config(
            latent_channels=latent_channels or channels,
            block_out_channels=block_out_channels or diffusers_levels(config),
            latents_mean=latents_mean or [0.0] * channels,
            latents_std=latents_std or [1.0] * channels,
        )

    def encode(self, x: torch.Tensor) -> _Encoded:
        """Encode to the raw latent (or the quantized one without a raw)."""
        out = self.model.bottleneck(self.model.encode(x))

        # diffusers callers read `.latent_dist.mean`; a vq has no raw, so z stands in
        latent = out.z if out.raw is None else out.raw
        return _Encoded(_Posterior(latent))

    def decode(self, z: torch.Tensor) -> _Decoded:
        """Decode a latent to the output signal."""
        return _Decoded(self.model.head(self.model.decode(z)))
