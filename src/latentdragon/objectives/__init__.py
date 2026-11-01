"""Objectives, paired with heads."""

from latentdragon.objectives.reconstruction import (
    HEAD_OBJECTIVES,
    OBJECTIVE,
    ReconstructionObjective,
)

__all__ = [
    # registries and head pairing
    "HEAD_OBJECTIVES",
    "OBJECTIVE",
    # concrete classes
    "ReconstructionObjective",
]
