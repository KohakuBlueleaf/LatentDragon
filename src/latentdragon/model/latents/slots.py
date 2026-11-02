"""One latent def per slot, composed: e.g. a continuous keyframe slot next to a vq motion slot."""

from torch import nn

from latentdragon.model.core.contracts import LATENT, Encoded, LatentDef
from latentdragon_common.registry import build


@LATENT.register("slots")
class SlotLatents(LatentDef):
    """Independent defs side by side; `defs` maps a name to each def's spec.

    Each def keeps its own slot names. Losses and logs are prefixed with the def's name, so
    `{"key": kl, "motion": vq}` reports `key/kl` and `motion/vq`.
    """

    def __init__(self, defs: dict[str, str | dict]):
        super().__init__()
        self.defs = nn.ModuleDict(
            {name: build(spec, LATENT) for name, spec in defs.items()}
        )
        self.inputs = tuple(s for d in self.defs.values() for s in d.inputs)
        self.outputs = tuple(s for d in self.defs.values() for s in d.outputs)
        self.decoder_slots = tuple(
            s for d in self.defs.values() for s in d.decoder_slots
        )

    def serialize(self, slots: dict, offset: int = 0) -> Encoded:
        """Every def on its own slots, merged."""
        merged = Encoded(latent={})
        for name, part in self.defs.items():
            out = part.serialize(slots, offset)
            merged.latent.update(out.latent)
            merged.losses.update({f"{name}/{k}": v for k, v in out.losses.items()})
            merged.logs.update({f"{name}/{k}": v for k, v in out.logs.items()})
            merged.extras[name] = out
        return merged

    def deserialize(self, latent: dict) -> dict:
        """Every def's decoder slots, merged."""
        slots = {}
        for part in self.defs.values():
            slots.update(part.deserialize(latent))
        return slots

    def pack(self, latent: dict) -> dict:
        """Every def's packed slots, merged."""
        packed = {}
        for part in self.defs.values():
            packed.update(part.pack(latent))
        return packed

    def raw(self, encoded: Encoded) -> dict:
        """Every def's continuous view, merged."""
        view = {}
        for name, part in self.defs.items():
            view.update(part.raw(encoded.extras[name]))
        return view

    def bind_kernels(self, device) -> None:
        for part in self.defs.values():
            part.bind_kernels(device)

    def track_stats(self, on: bool) -> None:
        for part in self.defs.values():
            part.track_stats(on)
