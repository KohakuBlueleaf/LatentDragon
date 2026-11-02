"""`AEArchConfig`, the named architecture presets and `build_autoencoder`."""

from dataclasses import dataclass, replace
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


KOHAKU_VQ = {
    "name": "vq",
    "quantizer": {
        "name": "kohaku",
        "n_embed": 65536,
        "base_dim": 16,
        "partition_mode": "hyper",
        "mlp_projector": True,
        "mlp_hidden_dim": 2048,
        "mlp_act": "silu",
        "separated_proj": False,
        "trainable_base": True,
        "dim_scale": True,
        "beta": 0.25,
        "ivq_loss_weight": 1e-3,
    },
}

F8_MLP_ENCODER = ((2, 256, 512, 2), (2, 512, 1024, 2), (2, 512, 1024, 2))
F16_KVQ_ENCODER = ((4, 256, 512, 2), (2, 256, 512, 2), (2, 512, 1024, 2))
F8_SD_TRUNK = ((1, 512, 1), (2, 512, 4), (2, 512, 3), (2, 256, 3), (1, 128, 3))
F16_FLAT_TRUNK = ((2, 512, 5), (2, 256, 3), (2, 256, 3), (2, 128, 3), (1, 128, 3))
SD_TRUNK_ATTN = (0, 1, 0, 0, 0)

PRESETS: dict[str, AEArchConfig] = {
    "kvq_f16c256": AEArchConfig(
        latent_dim=256,
        encoder_blocks=F16_KVQ_ENCODER,
        bottleneck=KOHAKU_VQ,
        trunk_blocks=(
            (1, 512, 1),
            (2, 512, 4),
            (2, 256, 3),
            (2, 256, 3),
            (2, 128, 3),
            (1, 128, 3),
        ),
        trunk_attn=(0, 1, 0, 0, 0, 0),
    ),
    "kvq_f16c256_production": AEArchConfig(
        latent_dim=256,
        encoder_blocks=F16_KVQ_ENCODER,
        bottleneck=KOHAKU_VQ,
        trunk="mlp",
        trunk_blocks=(
            (1, 64, 128, 2),
            (2, 128, 256, 2),
            (2, 256, 512, 2),
            (2, 512, 1024, 4),
            (2, 1024, 1024, 4),
        ),
    ),
    "kvq_f16c256_stylegan2": AEArchConfig(
        latent_dim=256,
        encoder_blocks=F16_KVQ_ENCODER,
        bottleneck=KOHAKU_VQ,
        trunk_blocks=F16_FLAT_TRUNK,
    ),
    "sdvae_f8c4": AEArchConfig(
        encoder="conv",
        encoder_blocks=(
            (1, 128, 2),
            (2, 256, 2),
            (2, 512, 2),
            (2, 512, 3),
            (1, 512, 1),
        ),
        encoder_attn=(0, 0, 0, 1, 0),
        encoder_norm="group32",
        bottleneck="kl",
        trunk_blocks=F8_SD_TRUNK,
        trunk_attn=SD_TRUNK_ATTN,
        trunk_norm="group32",
    ),
    "latentmaid_f8c4": AEArchConfig(
        encoder_blocks=F8_MLP_ENCODER,
        trunk_blocks=((2, 512, 5), (2, 512, 3), (2, 256, 3), (1, 128, 3)),
    ),
    "latentmaid_0222_f8c4": AEArchConfig(
        encoder_blocks=F8_MLP_ENCODER,
        bottleneck="kl",
        trunk_blocks=F8_SD_TRUNK,
        trunk_attn=SD_TRUNK_ATTN,
        trunk_norm="group32",
    ),
    "latentmaid_0222_better_dec_f8c4": AEArchConfig(
        encoder_blocks=F8_MLP_ENCODER,
        bottleneck="kl",
        trunk_blocks=((2, 512, 4), (2, 512, 2), (2, 256, 2), (1, 128, 2)),
    ),
    "latentmaid_0222_f16c64n16": AEArchConfig(
        latent_dim=64,
        encoder_blocks=((4, 128, 512, 2), (2, 256, 1024, 2), (2, 512, 1024, 2)),
        bottleneck={"name": "nested", "inner": "kl", "latent_chs": (8, 16, 32, 64)},
        trunk_blocks=F16_FLAT_TRUNK,
    ),
    "video_f16t6c32": AEArchConfig(
        domain="video",
        latent_dim=32,
        encoder="video",
        encoder_blocks=((3, 8, 128, 512, 2), (2, 2, 256, 1024, 2)),
        trunk="video",
        trunk_blocks=(
            (1, 1, 64, 128, 1),
            (2, 2, 128, 256, 2),
            (3, 2, 256, 512, 4),
            (1, 2, 512, 1024, 6),
            (1, 2, 512, 1024, 8),
        ),
    ),
}


def get_preset(name: str, **overrides) -> AEArchConfig:
    """A named preset with fields replaced by `overrides`."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset '{name}'; known: {', '.join(sorted(PRESETS))}")
    return replace(PRESETS[name], **overrides)


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
