"""`AEArchConfig`, the fields of one autoencoder, and `build_autoencoder`."""

from dataclasses import dataclass
from typing import Any

from latentdragon.bottleneck import BOTTLENECK
from latentdragon.domain import DOMAIN
from latentdragon.model.autoencoder import Autoencoder
from latentdragon.model.encoders import ENCODER
from latentdragon.model.heads import HEAD
from latentdragon.model.trunks import TRUNK
from latentdragon_common.registry import build


@dataclass(frozen=True)
class AEArchConfig:
    """One autoencoder shape; component fields take any `build` choice."""

    domain: str = "image"
    image_dim: int = 3
    latent_dim: int = 4
    encoder: Any = "mlp"
    encoder_blocks: tuple = ()
    encoder_attn: tuple | None = None
    encoder_norm: str = "rms"
    bottleneck: Any = "identity"
    trunk: Any = "conv"
    trunk_blocks: tuple = ()
    trunk_attn: tuple | None = None
    trunk_norm: str = "rms"
    head: Any = "regression"


def build_autoencoder(cfg: AEArchConfig) -> Autoencoder:
    """Build every part; the encoder emits what the bottleneck asks for."""
    # Domain and bottleneck first: the bottleneck sets the encoder width
    domain = build(cfg.domain, DOMAIN)
    bottleneck = build(cfg.bottleneck, BOTTLENECK, latent_dim=cfg.latent_dim)

    encoder = build(
        cfg.encoder,
        ENCODER,
        image_dim=cfg.image_dim,
        latent_dim=cfg.latent_dim,
        block_configs=cfg.encoder_blocks,
        attn_block_configs=cfg.encoder_attn,
        norm_layer=cfg.encoder_norm,
        out_channels=bottleneck.in_channels(cfg.latent_dim),
    )
    bottleneck.init_encoder_out(encoder.out_layer, cfg.latent_dim)

    trunk = build(
        cfg.trunk,
        TRUNK,
        latent_dim=cfg.latent_dim,
        block_configs=cfg.trunk_blocks,
        attn_block_configs=cfg.trunk_attn,
        norm_layer=cfg.trunk_norm,
    )

    # Encoder and trunk must agree on the total scale
    if (encoder.spatial_scale, encoder.temporal_scale) != (
        trunk.spatial_scale,
        trunk.temporal_scale,
    ):
        raise ValueError(
            f"encoder scale {encoder.spatial_scale}x{encoder.temporal_scale} differs "
            f"from trunk scale {trunk.spatial_scale}x{trunk.temporal_scale}"
        )

    head = build(
        cfg.head, HEAD, dim=trunk.out_dim, out_dim=cfg.image_dim, **trunk.head_form
    )
    return Autoencoder(encoder, bottleneck, trunk, head, domain)
