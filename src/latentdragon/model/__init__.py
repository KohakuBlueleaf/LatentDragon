"""Autoencoder models: the slot contracts, their parts and the build.

Importing this package registers every built-in encoder, latent def, decoder and head.
"""

from latentdragon.model import families, heads, latents
from latentdragon.model.core import (
    DECODER,
    ENCODER,
    HEAD,
    LATENT,
    Autoencoder,
    Encoded,
    LatentDef,
    Part,
    TensorPart,
    build_autoencoder,
    tree,
)
from latentdragon.model.nn import BLOCK, NORM, Reach
from latentdragon.model.runtime.compile import (
    allow_donated_buffers,
    compile_parts,
    compile_wrapper,
    configure,
)
from latentdragon.model.runtime.recompute import apply_recompute

__all__ = [
    # registries
    "BLOCK",
    "DECODER",
    "ENCODER",
    "HEAD",
    "LATENT",
    "NORM",
    # contracts and records
    "Encoded",
    "LatentDef",
    "Part",
    "Reach",
    "TensorPart",
    # composition
    "Autoencoder",
    "build_autoencoder",
    # runtime
    "allow_donated_buffers",
    "apply_recompute",
    "compile_parts",
    "compile_wrapper",
    "configure",
    # registering subpackages and slot trees
    "families",
    "heads",
    "latents",
    "tree",
]
