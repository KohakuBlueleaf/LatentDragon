"""Reconstruction objective over a composed sensecraft loss."""

import torch
from sensecraft import Loss
from torch import Tensor

from latentdragon_common.registry import Registry

OBJECTIVE = Registry("objective")
HEAD_OBJECTIVES = {"regression": "reconstruction"}
DIMS = {"2d": 2, "3d": 3}


@OBJECTIVE.register("reconstruction")
class ReconstructionObjective:
    """Weighted sensecraft terms of `(x, x_hat)`; the loss networks are frozen."""

    def __init__(self, loss_configs: list, mode: str = "2d"):
        self.loss = (
            Loss(list(loss_configs), dims=DIMS[mode]).requires_grad_(False).eval()
        )

    def compile(self, **kwargs) -> None:
        """Compile every term that holds a network; parameter-free terms stay eager."""
        for part in self.loss.parts.values():
            if any(True for _ in part.parameters()):
                part.compile(**kwargs)

    def to(
        self, device, weights: torch.dtype | None = None
    ) -> "ReconstructionObjective":
        """Move the terms; `weights` holds the frozen feature nets in that dtype."""
        self.loss.prepare(device, weights)
        return self

    def __call__(self, x: Tensor, x_hat: Tensor) -> tuple[Tensor, dict]:
        """`(nll, term logs)` of the reconstruction `x_hat` of `x`."""
        out = self.loss(x_hat, x)
        logs = {name: value.detach() for name, value in out.terms.items()}
        return out.total, {**logs, "loss": out.total.detach()}

    def seeded(
        self, x: Tensor, x_hat: Tensor, scale: float
    ) -> tuple[Tensor, Tensor, dict]:
        """`(d nll / d x_hat, nll, term logs)`, with no graph kept through the loss nets."""
        out = self.loss.value_and_seed(x_hat, x, scale)
        return out.seed / scale, out.total, {**out.terms, "loss": out.total}
