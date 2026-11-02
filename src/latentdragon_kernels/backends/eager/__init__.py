"""Eager reference backend: valid on every device."""

from latentdragon_kernels.backends.eager import adam16, mlp, quantize

__all__ = ["adam16", "mlp", "quantize"]
