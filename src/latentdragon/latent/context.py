"""What latent views and terms can read."""

from dataclasses import dataclass, field
from typing import Any

from latentdragon.model import Encoded


@dataclass
class LatentCtx:
    """One micro-batch: data slots, the encoded record, requested teacher features.

    `fake` and `seed` are the reconstruction slots and their gradient seeds, for weightings
    that compare gradients. `states` holds the main pass's slot-boundary values by the part
    that wrote them: `encoder` (before the latent def), `latent`, `decoder` (the output latent
    the head reads).
    """

    x: dict[str, Any]
    encoded: Encoded
    step: int
    teacher: dict[str, Any] = field(default_factory=dict)
    fake: dict[str, Any] | None = None
    seed: dict[str, Any] | None = None
    states: dict[str, Any] = field(default_factory=dict)
