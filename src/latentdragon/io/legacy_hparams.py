"""Hparams conversion between legacy configs and `ARCH`, through the flat arch fields."""

from typing import Any

from latentdragon.io.arch_json import flat_fields, upgrade_arch

NORMS = {
    "latentmaid.model.modules.RMSNorm": "rms",
    "latentmaid.model.modules.LayerNorm": "layer",
    "latentmaid.model.modules.GroupNorm32": "group32",
    "latentmaid.model.modules.GruopNorm32": "group32",
}
LEGACY_NORMS = {
    "rms": "latentmaid.model.modules.RMSNorm",
    "layer": "latentmaid.model.modules.LayerNorm",
    "group32": "latentmaid.model.modules.GroupNorm32",
}
OFF_VALUES = {
    "addon_mlp": 0,
    "use_grouped_swiglu": False,
    "t_mix_first": False,
}
# The encoder's latent norm is the vq latent def's `normalize="rms"`, read separately
LATENT_NORM_KEYS = ("latent_norm", "latent_norm_partitions")
DROPPED = ("diff_head_config", "variational", "logvar_clamp")
UNPORTED_VQ = {
    "softbwd_mode": None,
    "diveq": False,
    "sf_diveq": False,
    "sf_ste": False,
    "codebook_mixing": None,
    "reg_loss": None,
    "latent_norm": False,
}
VQ_ALWAYS_DROPPED = (
    "softbwd_temperature",
    "diveq_sigma",
    "codebook_mixing_factor",
    "codebook_mixing_dim_factor",
    "reg_weight",
    "reg_loss_on_base",
    "reg_weight_on_base",
    "usage_ema_decay",
    "legacy",
)


def _norm(name: Any) -> str:
    return NORMS[name] if name in NORMS else "rms"


def _blocks(config: dict) -> list:
    return [list(row) for row in config["block_configs"]]


def _attn(config: dict) -> list | None:
    value = config.get("attn_block_configs")
    return list(value) if value else None


def _refuse_active(config: dict, label: str) -> None:
    for key, off in OFF_VALUES.items():
        if key in config and config[key] != off:
            raise ValueError(f"{label}.{key}={config[key]!r} is not ported")


def legacy_quantizer(vq_mode: str, vq_config: dict) -> dict:
    """A quantizer spec from `vq_mode` and `vq_config`."""
    if vq_mode not in ("kohaku", "naive"):
        raise ValueError(f"vq_mode {vq_mode!r} is not ported")
    spec = dict(vq_config)
    for key, off in UNPORTED_VQ.items():
        if spec.pop(key, off) != off:
            raise ValueError(f"vq_config.{key} is not ported")
    if spec.get("legacy", False):
        raise ValueError("NaiveVQ legacy=True is not ported")
    for key in VQ_ALWAYS_DROPPED:
        spec.pop(key, None)
    non_shared = spec.pop("non_shared", None)
    mode = spec.pop("partition_mode", None)
    if non_shared is not None:
        mode = "indep" if non_shared else None
    if vq_mode == "kohaku" or mode is not None:
        spec["partition_mode"] = mode or "shared"
    return {"name": vq_mode, **spec}


def _latent_norm(enc: dict, quantizer: dict | None) -> str | None:
    """`"rms"` when the legacy encoder normalized its output per vq partition, else None."""
    if not enc.get("latent_norm", False):
        return None
    if quantizer is None:
        raise ValueError("enc_config.latent_norm without a vq is not ported")
    partitions = enc.get("latent_norm_partitions", 1)
    if partitions != quantizer.get("partitions", 1):
        raise ValueError(
            f"latent_norm_partitions {partitions} differs from the vq partitions; not ported"
        )
    return "rms"


def _bottleneck(hparams: dict) -> Any:
    """The bottleneck spec (vq, kl or identity, optionally nested) of legacy hparams."""
    variational = hparams.get("variational", False)
    if hparams.get("vq_mode") is not None:
        quantizer = legacy_quantizer(hparams["vq_mode"], hparams.get("vq_config") or {})
        vq = {"name": "vq", "quantizer": quantizer}
        normalize = _latent_norm(hparams["enc_config"], quantizer)
        if normalize is not None:
            vq["normalize"] = normalize
        parts = hparams.get("nested_parts")
        if not parts:
            return vq
        return {
            "name": "nested",
            "inner": vq,
            "unit": "partition",
            "latent_chs": list(parts),
            "jitter": 1,
        }
    enc = hparams["enc_config"]
    _latent_norm(enc, None)
    inner = "identity"
    if variational:
        clamp = enc.get("logvar_clamp")
        inner = (
            {"name": "kl", "logvar_clamp": tuple(clamp)} if clamp else {"name": "kl"}
        )
    chs = hparams.get("latent_chs")
    if not chs:
        return inner
    return {
        "name": "nested",
        "inner": inner,
        "latent_chs": list(chs),
        "ratio": hparams.get("latent_chs_ratio") or None,
    }


def from_legacy_hparams(hparams: dict) -> dict:
    """An `ARCH` from a checkpoint's `hyper_parameters`."""
    enc, dec = hparams["enc_config"], hparams["dec_config"]
    _refuse_active(enc, "enc_config")
    _refuse_active(dec, "dec_config")
    if dec.get("latent_norm", False):
        raise ValueError("dec_config.latent_norm is not ported")
    video = bool(hparams.get("is_video", False))
    flat = dict(
        domain="video" if video else "image",
        image_dim=enc.get(
            "image_dim", enc.get("input_dim", dec.get("out_channels", 3))
        ),
        latent_dim=enc["latent_dim"],
        encoder="video" if video else enc.get("encoder_type", "mlp"),
        encoder_blocks=_blocks(enc),
        encoder_attn=_attn(enc),
        encoder_norm=_norm(enc.get("norm_layer")),
        bottleneck=_bottleneck(hparams),
        trunk="video" if video else dec["decoder_type"],
        trunk_blocks=_blocks(dec),
        trunk_attn=_attn(dec),
        trunk_norm=_norm(dec.get("norm_layer")),
    )
    return upgrade_arch(flat)


def _legacy_config(blocks, attn, norm, cfg: dict, variational: bool) -> dict:
    """One legacy `enc_config` or `dec_config` dict from the flat fields `cfg`."""
    config = {
        "image_dim": cfg["image_dim"],
        "latent_dim": cfg["latent_dim"],
        "block_configs": [list(row) for row in blocks],
        "norm_layer": LEGACY_NORMS[norm],
    }
    if attn:
        config["attn_block_configs"] = list(attn)
    if variational:
        config["variational"] = True
    return config


def _name(spec: Any) -> str:
    return spec if isinstance(spec, str) else spec["name"]


def to_legacy_hparams(arch: dict) -> dict:
    """`enc_config`, `dec_config` and VQ fields of the legacy layout."""
    cfg = flat_fields(arch)
    name = _name(cfg["bottleneck"])
    nested = cfg["bottleneck"] if name == "nested" else None
    if nested is not None:
        name = _name(nested["inner"])
    variational = name == "kl"
    enc = _legacy_config(
        cfg["encoder_blocks"],
        cfg["encoder_attn"],
        cfg["encoder_norm"],
        cfg,
        variational,
    )
    enc["encoder_type"] = cfg["encoder"]
    dec = _legacy_config(
        cfg["trunk_blocks"], cfg["trunk_attn"], cfg["trunk_norm"], cfg, False
    )
    dec["decoder_type"] = cfg["trunk"]
    out = {
        "enc_config": enc,
        "dec_config": dec,
        "is_video": cfg["domain"] == "video",
        "variational": variational,
    }
    if nested is not None and name == "vq":
        if nested.get("unit") != "partition":
            raise ValueError("nested around vq exports only with unit='partition'")
        out["nested_parts"] = list(nested["latent_chs"])
    elif nested is not None:
        out["latent_chs"] = list(nested["latent_chs"])
        out["latent_chs_ratio"] = nested.get("ratio")
    if name == "vq":
        vq = nested["inner"] if nested is not None else cfg["bottleneck"]
        spec = dict(vq["quantizer"])
        if vq.get("normalize") == "rms":
            enc["latent_norm"] = True
            enc["latent_norm_partitions"] = spec.get("partitions", 1)
        out["vq_mode"] = spec.pop("name")
        if spec.get("partition_mode") == "shared":
            spec["partition_mode"] = None
        out["vq_config"] = spec
    return out
