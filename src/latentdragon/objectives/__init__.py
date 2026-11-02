"""Objectives over data slots; a head names the one it pairs with."""

from latentdragon.objectives.reconstruction import (
    OBJECTIVE,
    ReconstructionObjective,
    per_slot,
)

__all__ = [
    # registries
    "OBJECTIVE",
    # concrete classes
    "ReconstructionObjective",
    # functions
    "per_slot",
]
