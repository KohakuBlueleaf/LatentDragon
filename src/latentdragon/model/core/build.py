"""`build_autoencoder(ARCH)`: every part from its registry, then the slot wiring check.

    ARCH = {
        "domain": {"name": "image"},
        "encoder": {"name": ..., ...},
        "latent": {"name": ..., ...},
        "decoder": {"name": ..., ...},
        "head": {"name": ..., ...},
    }

Each entry is a registry choice: a name, a dotted path or `{"name": ..., **kwargs}`. The
framework passes no arguments of its own; every width and slot name is in the entry.
"""

from latentdragon.domain import DOMAIN
from latentdragon.model.core.autoencoder import Autoencoder
from latentdragon.model.core.contracts import DECODER, ENCODER, HEAD, LATENT
from latentdragon_common.registry import build

PARTS = ("domain", "encoder", "latent", "decoder", "head")
DEFAULTS = {"domain": "image", "latent": "identity"}


def wiring_problems(domain, encoder, latent, decoder, head) -> list[str]:
    """Slots a part reads that the part before it does not write."""
    links = (
        ("encoder", encoder.inputs, "domain", domain.slots),
        ("latent", latent.inputs, "encoder", encoder.outputs),
        ("decoder", decoder.inputs, "latent", latent.decoder_slots),
        ("head", head.inputs, "decoder", decoder.outputs),
        ("domain", head.outputs, "head", domain.slots),
    )
    problems = []
    for reader, wanted, writer, written in links:
        missing = [slot for slot in wanted if slot not in written]
        if missing:
            problems.append(
                f"{reader} reads {missing}, which {writer} does not write "
                f"(it writes {list(written)})"
            )
    return problems


def scale_problems(encoder, decoder) -> list[str]:
    """Declared encoder and decoder scales that disagree."""
    keys = ("spatial_scale", "temporal_scale")
    if not all(hasattr(part, key) for part in (encoder, decoder) for key in keys):
        return []
    down = tuple(getattr(encoder, key) for key in keys)
    up = tuple(getattr(decoder, key) for key in keys)
    if down == up:
        return []
    return [
        f"encoder scale {down[0]}x{down[1]} differs from decoder scale {up[0]}x{up[1]}"
    ]


def build_autoencoder(arch: dict) -> Autoencoder:
    """Build every part of `arch` and check that each reads what the one before writes."""
    unknown = set(arch) - set(PARTS)
    if unknown:
        raise KeyError(f"ARCH has unknown parts {sorted(unknown)}; parts are {PARTS}")
    spec = {part: arch.get(part, DEFAULTS.get(part)) for part in PARTS}
    for part in ("encoder", "decoder", "head"):
        if spec[part] is None:
            raise KeyError(f"ARCH needs a {part}")

    domain = build(spec["domain"], DOMAIN)
    encoder = build(spec["encoder"], ENCODER)
    latent = build(spec["latent"], LATENT)
    decoder = build(spec["decoder"], DECODER)
    head = build(spec["head"], HEAD)

    problems = wiring_problems(domain, encoder, latent, decoder, head)
    problems += scale_problems(encoder, decoder)
    if problems:
        raise ValueError("; ".join(problems))

    # A latent def may prepare the encoder layer feeding it, e.g. kl zeroes the logvar half
    if hasattr(latent, "init_from_encoder") and hasattr(encoder, "last_layer"):
        latent.init_from_encoder(encoder.last_layer)
    return Autoencoder(domain, encoder, latent, decoder, head)
