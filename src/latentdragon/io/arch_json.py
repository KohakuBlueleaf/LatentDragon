"""`ARCH` as JSON values, and the flat arch format of older runs and exports.

The flat format named every field of the built-in families at the top level (`latent_dim`,
`encoder_blocks`, `trunk`, `bottleneck`, ...). `upgrade_arch` turns it into the part form;
`flat_fields` goes back for converters whose target format is flat.
"""

import json

FAMILIES = {"mlp": "latentmaid", "conv": "sdvae", "video": "latentmaid_video"}
FLAT_NAMES = {new: old for old, new in FAMILIES.items()}
FLAT_KEYS = ("latent_dim", "encoder_blocks", "trunk", "bottleneck", "trunk_blocks")


def arch_to_json(arch: dict) -> dict:
    """The arch as JSON values: tuples become lists."""
    return json.loads(json.dumps(arch))


def is_flat(values: dict) -> bool:
    """True for the flat format of older runs and exports."""
    return any(key in values for key in FLAT_KEYS)


def _width_doubled(bottleneck) -> bool:
    """Whether a flat bottleneck spec reads twice the latent channels (kl, or nested kl)."""
    match bottleneck:
        case "kl" | {"name": "kl"}:
            return True
        case {"name": "nested", "inner": inner}:
            return _width_doubled(inner)
    return False


def _latent(bottleneck, latent_dim: int):
    """A flat bottleneck spec as a latent def spec."""
    match bottleneck:
        case str():
            return bottleneck
        case {"name": "vq" | "nested"}:
            spec = {**bottleneck, "dim": latent_dim}
            if "inner" in spec:
                spec["inner"] = _latent(spec["inner"], latent_dim)
            return spec
    return dict(bottleneck)


def _decoder_width(family: str, stages: list) -> int:
    """The feature width a flat decoder hands its head."""
    if family == "sdvae":
        return stages[-1][1]
    first = stages[0]
    scale, hidden = (first[1], first[2]) if family == "latentmaid_video" else first[:2]
    return hidden // scale**2


def _head(family: str, width: int, image_dim: int, norm: str) -> dict:
    """The head a flat decoder family implied."""
    head = {"name": "conv", "in_channels": width, "out_channels": image_dim}
    if family == "sdvae":
        return {**head, "kernel_size": 3, "norm": norm, "act": "mish"}
    if family == "latentmaid_video":
        return {**head, "kernel_size": 1, "video": True}
    return {**head, "kernel_size": 1}


def upgrade_arch(values: dict) -> dict:
    """A flat arch in the part form; a part-form arch is returned unchanged."""
    if not is_flat(values):
        return dict(values)
    image_dim = values.get("image_dim", 3)
    latent_dim = values.get("latent_dim", 4)
    bottleneck = values.get("bottleneck", "identity")
    encoder = FAMILIES[values.get("encoder", "mlp")]
    decoder = FAMILIES[values.get("trunk", "conv")]
    encoder_stages = [list(row) for row in values["encoder_blocks"]]
    decoder_stages = [list(row) for row in values["trunk_blocks"]]
    encoder_norm = values.get("encoder_norm", "rms")
    decoder_norm = values.get("trunk_norm", "rms")
    width = 2 * latent_dim if _width_doubled(bottleneck) else latent_dim

    def part(name, stages, attn, norm, **sizes):
        spec = {"name": name, **sizes, "stages": stages, "norm": norm}
        if name != "latentmaid_video":
            spec["attn"] = list(attn) if attn else None
        return spec

    return {
        "domain": values.get("domain", "image"),
        "encoder": part(
            encoder,
            encoder_stages,
            values.get("encoder_attn"),
            encoder_norm,
            in_channels=image_dim,
            out_channels=width,
        ),
        "latent": _latent(bottleneck, latent_dim),
        "decoder": part(
            decoder,
            decoder_stages,
            values.get("trunk_attn"),
            decoder_norm,
            in_channels=latent_dim,
        ),
        "head": _head(
            decoder, _decoder_width(decoder, decoder_stages), image_dim, decoder_norm
        ),
    }


def _flat_bottleneck(latent):
    """A latent def spec back in the flat bottleneck form."""
    match latent:
        case str():
            return latent
        case {"name": "vq" | "nested"}:
            spec = {
                k: v for k, v in latent.items() if k not in ("dim", "input", "output")
            }
            if "inner" in spec:
                spec["inner"] = _flat_bottleneck(spec["inner"])
            return spec
    return dict(latent)


def _name(spec) -> str:
    return spec if isinstance(spec, str) else spec["name"]


def flat_fields(arch: dict) -> dict:
    """The flat fields of a part-form arch built only from the built-in families.

    Raises for any part the flat format cannot hold.
    """
    encoder, decoder = arch["encoder"], arch["decoder"]
    names = {_name(encoder), _name(decoder)}
    if not names <= set(FLAT_NAMES):
        raise ValueError(
            f"the flat format holds only {sorted(FLAT_NAMES)}, not {names}"
        )
    latent = arch.get("latent", "identity")
    if _name(latent) not in ("identity", "kl", "vq", "nested"):
        raise ValueError(f"the flat format cannot hold latent {_name(latent)!r}")
    return {
        "domain": _name(arch.get("domain", "image")),
        "image_dim": encoder.get("in_channels", 3),
        "latent_dim": decoder["in_channels"],
        "encoder": FLAT_NAMES[encoder["name"]],
        "encoder_blocks": [list(row) for row in encoder["stages"]],
        "encoder_attn": encoder.get("attn"),
        "encoder_norm": encoder.get("norm", "rms"),
        "bottleneck": _flat_bottleneck(latent),
        "trunk": FLAT_NAMES[decoder["name"]],
        "trunk_blocks": [list(row) for row in decoder["stages"]],
        "trunk_attn": decoder.get("attn"),
        "trunk_norm": decoder.get("norm", "rms"),
    }
