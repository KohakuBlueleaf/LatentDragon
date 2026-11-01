"""Triton backend: an ImportError here marks the backend unavailable."""

from latentdragon_kernels.backends.triton import mlp, quantize

__all__ = ["mlp", "quantize"]
