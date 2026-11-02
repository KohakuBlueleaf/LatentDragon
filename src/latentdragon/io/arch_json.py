"""`AEArchConfig` to and from JSON values."""

import dataclasses
import json

from latentdragon.model import AEArchConfig


def arch_to_json(arch: AEArchConfig) -> dict:
    """The arch as JSON values."""
    return json.loads(json.dumps(dataclasses.asdict(arch)))


def diffusers_levels(arch: AEArchConfig) -> list[int]:
    """Stage widths padded to one entry per power-of-two scale (`block_out_channels`)."""
    scale = 1
    for row in arch.encoder_blocks:
        scale *= row[0]

    widths = [row[1] for row in arch.encoder_blocks]
    return (widths + widths[-1:] * scale)[: scale.bit_length()]


def arch_from_json(values: dict) -> AEArchConfig:
    """An arch from JSON or config values; block rows become tuples, absent fields default."""
    fields = dict(values)
    for key in ("encoder_blocks", "trunk_blocks"):
        if key in fields:
            fields[key] = tuple(tuple(row) for row in fields[key])
    for key in ("encoder_attn", "trunk_attn"):
        if key in fields:
            fields[key] = tuple(fields[key]) if fields[key] else None
    return AEArchConfig(**fields)
