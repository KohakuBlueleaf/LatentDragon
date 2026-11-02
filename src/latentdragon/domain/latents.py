"""Latents as RGB for sample grids: the top three principal components of the channels."""

import torch
from torch import Tensor

QUANTILES = (0.05, 0.95)


def pca_rgb(latents: Tensor, eps: float = 1e-8) -> Tensor:
    """`(B, C, *spatial)` latents as `(B, 3, *spatial)` in [0, 1].

    Projects the channels onto their top three principal components over every position of the
    batch, then maps each component's 5% to 95% quantile range onto [0, 1].
    """
    batch, channels, *spatial = latents.shape
    with torch.autocast(latents.device.type, enabled=False):
        points = latents.float().movedim(1, -1).reshape(-1, channels)
        centered = points - points.mean(dim=0, keepdim=True)
        centered = centered / (points.std() + eps)

        # Eigenvectors come in ascending order, so the last three are the top components
        covariance = centered.T @ centered / (centered.shape[0] - 1)
        _, vectors = torch.linalg.eigh(covariance)
        rgb = centered @ vectors[:, -3:]

        low = torch.quantile(rgb, QUANTILES[0], dim=0)
        high = torch.quantile(rgb, QUANTILES[1], dim=0)
        rgb = ((rgb - low) / (high - low + eps)).clamp(0, 1)
    return rgb.reshape(batch, *spatial, 3).movedim(-1, 1)
