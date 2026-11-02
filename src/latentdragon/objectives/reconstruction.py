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

    def compile(self, **kwargs) -> None:
        """Compile every loss network; parameter-free losses stay eager."""
        for module in self.loss.loss_modules.values():
            if any(True for _ in module.parameters()):
                module.compile(**kwargs)

    def to(
        self, device, weights: torch.dtype | None = None
    ) -> "ReconstructionObjective":
        """Move the loss networks; `weights` holds their frozen weights in that dtype.

        Under autocast a weight is cast on every forward, and frozen weights round the same
        way each time, so holding them in the compute dtype removes the casts and nothing else.
        """
        self.loss.to(device)
        if weights is not None:
            for param in self.loss.parameters():
                param.data = param.data.to(weights)
        return self

    def __call__(self, x: torch.Tensor, x_hat: torch.Tensor) -> tuple:
        terms = self.loss(x, x_hat, return_dict=True)
        nll = terms["loss"]
        return nll, {k: v.detach() for k, v in terms.items()}
