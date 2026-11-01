"""Vector quantization on one partition x residual grid."""

# Codebook sources
from latentdragon.vq.codebook import CODEBOOK, DirectCodebook, ProjectedCodebook

# Grid quantizer and its registry
from latentdragon.vq.grid import GridQuantizer
from latentdragon.vq.quantizers import QUANTIZER

__all__ = [
    # registries
    "CODEBOOK",
    "QUANTIZER",
    # concrete classes
    "DirectCodebook",
    "GridQuantizer",
    "ProjectedCodebook",
]
