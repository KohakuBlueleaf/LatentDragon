"""Reconstruction objective over a composed SenseCraft loss."""

import torch
from sensecraft.loss import SenseCraftLoss

from latentdragon_common.registry import Registry

OBJECTIVE = Registry("objective")
HEAD_OBJECTIVES = {"regression": "reconstruction"}


@OBJECTIVE.register("reconstruction")
class ReconstructionObjective:
    """`(x, x_hat) -> (nll, logs)`; the loss networks are frozen and stay in eval."""

    def __init__(self, loss_configs: list, mode: str = "2d"):
        self.loss = (
            SenseCraftLoss(list(loss_configs), mode=mode, return_dict=True)
            .requires_grad_(False)
            .eval()
        )

    def to(self, device) -> "ReconstructionObjective":
        self.loss.to(device)
        return self

    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor) -> tuple:
        terms = self.loss(x, x_hat, return_dict=True)
        nll = terms["loss"]
        return nll, {k: v.detach() for k, v in terms.items()}
