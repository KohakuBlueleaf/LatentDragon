"""Host-only training state."""

from dataclasses import asdict, dataclass


@dataclass
class TrainState:
    """Step and data position; everything here is a plain host value."""

    step: int = 0
    epoch: int = 0
    batches: int = 0

    def to_dict(self) -> dict:
        return asdict(self)
