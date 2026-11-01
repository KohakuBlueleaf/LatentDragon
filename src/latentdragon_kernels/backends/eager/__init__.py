"""Eager reference backend: valid on every device."""

from latentdragon_kernels.backends.eager import mlp, quantize

__all__ = ["mlp", "quantize"]
