"""Gemm backend: fused gradients through the card's BLAS, without holding activations."""

from latentdragon_kernels.backends.gemm import mlp

__all__ = ["mlp"]
