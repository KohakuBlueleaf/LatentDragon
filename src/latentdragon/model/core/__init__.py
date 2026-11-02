"""The standard IO: slot contracts, the slot tree helpers, the autoencoder and its build."""

from latentdragon.model.core import tree
from latentdragon.model.core.autoencoder import Autoencoder
from latentdragon.model.core.build import build_autoencoder, wiring_problems
from latentdragon.model.core.contracts import (
    DECODER,
    ENCODER,
    HEAD,
    LATENT,
    Encoded,
    LatentDef,
    Part,
    TensorPart,
)

__all__ = [
    # registries
    "DECODER",
    "ENCODER",
    "HEAD",
    "LATENT",
    # contracts
    "Encoded",
    "LatentDef",
    "Part",
    "TensorPart",
    # composition
    "Autoencoder",
    "build_autoencoder",
    "wiring_problems",
    # slot trees
    "tree",
]
