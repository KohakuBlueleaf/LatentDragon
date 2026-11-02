"""The identity head: decoder slots renamed to data slots."""

from torch import nn

from latentdragon.model.core.contracts import HEAD


@HEAD.register("identity")
class IdentityHead(nn.Module):
    """Renames decoder slots to data slots, for decoders that end in the signal.

    `slots` maps each decoder slot to the data slot it fills.
    """

    objective = "reconstruction"
    needs_pixels = False
    reach = 0

    def __init__(self, slots: dict[str, str] | None = None):
        super().__init__()
        self.slots = dict(slots or {"y": "x"})
        self.inputs = tuple(self.slots)
        self.outputs = tuple(self.slots.values())

    def forward(self, slots: dict) -> dict:
        """The decoder slots under their data slot names."""
        return {target: slots[source] for source, target in self.slots.items()}
