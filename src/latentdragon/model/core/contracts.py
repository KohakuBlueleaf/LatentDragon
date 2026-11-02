"""The standard IO of an autoencoder: named slots across every arrow.

    domain -> encoder -> latent.serialize -> latent -> latent.deserialize -> decoder -> head
                                                                                -> domain

Every value crossing an arrow is a dict of named slots, each a batch-first tensor or a pytree
of them. A part declares the slot names it reads (`inputs`) and writes (`outputs`); the
build checks that each part reads what the one before it writes. Shapes, counts and widths
inside a slot belong to the parts, never to the framework.

The latent def is the serdes between the two halves: `serialize` turns encoder slots into the
latent (plus its losses and logs), `deserialize` turns a latent, live or stored, into decoder
slots, and `pack` gives the storable form. Anything shared by both directions, such as a
codebook, lives in the latent def.

Everything else a part may offer is an optional capability, looked up by whoever needs it:
`last_layer` (adaptive weights), `reach`, `scales`, `forward_stream` (decode strategies),
`raw` and `eval_variants` (stats, previews, nested eval), `codebook_size` (code usage).
"""

from dataclasses import dataclass, field
from typing import Any

from torch import Tensor, nn

from latentdragon_common.registry import Registry

ENCODER = Registry("encoder")
LATENT = Registry("latent")
DECODER = Registry("decoder")
HEAD = Registry("head")


@dataclass
class Encoded:
    """What `serialize` returns.

    `latent` is what `deserialize` reads; while training it may hold forms only training
    needs, such as straight-through code vectors next to their indices. `losses` are live
    scalars the task weights by name, `logs` are detached scalars whose keys carry their log
    section, and `extras` is private to the latent def that made it.
    """

    latent: dict[str, Any]
    losses: dict[str, Tensor] = field(default_factory=dict)
    logs: dict[str, Tensor] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)


class Part(nn.Module):
    """One arrow of the chain: `forward(slots) -> slots`."""

    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    def forward(self, slots: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


class TensorPart(Part):
    """A part that reads one slot and writes one; `tensor` is its slot-local call."""

    def __init__(self, input: str, output: str):
        super().__init__()
        self.input = input
        self.output = output
        self.inputs = (input,)
        self.outputs = (output,)

    def forward(self, slots: dict[str, Any]) -> dict[str, Any]:
        """Map the input slot to the output slot."""
        return {self.output: self.tensor(slots[self.input])}

    def tensor(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class LatentDef(nn.Module):
    """The latent serdes: encoder slots to a latent, and a latent to decoder slots.

    `inputs` are the encoder slots it reads, `outputs` the latent slots it writes, and
    `decoder_slots` the slots `deserialize` hands the decoder.
    """

    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()
    decoder_slots: tuple[str, ...] = ()

    def serialize(self, slots: dict[str, Any], offset: int = 0) -> Encoded:
        """Encoder slots to the latent; `offset` is the global index of the first sample."""
        raise NotImplementedError

    def deserialize(self, latent: dict[str, Any]) -> dict[str, Any]:
        """A live or packed latent to decoder slots."""
        raise NotImplementedError

    def pack(self, latent: dict[str, Any]) -> dict[str, Any]:
        """The storable form of a latent; the latent itself by default."""
        return latent

    def raw(self, encoded: Encoded) -> dict[str, Any]:
        """The continuous view for stats and previews; the latent by default."""
        return encoded.latent

    def eval_variants(self) -> list:
        """Keys of latent variants evaluated separately; none by default."""
        return []

    def variant(self, latent: dict[str, Any], key) -> dict[str, Any]:
        """The latent reduced to variant `key`; only defs with variants implement it."""
        raise NotImplementedError

    def bind_kernels(self, device) -> None:
        """Resolve kernels for `device`; nothing by default."""

    def track_stats(self, on: bool) -> None:
        """Turn statistics that only feed logs on or off; nothing by default."""
