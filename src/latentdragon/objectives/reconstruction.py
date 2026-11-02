"""Reconstruction objective: sensecraft losses per data slot."""

import torch
from sensecraft import Loss
from torch import Tensor, nn

from latentdragon_common.registry import Registry

OBJECTIVE = Registry("objective")


def per_slot(loss_configs: list | dict, loss_dims: dict[str, int]) -> dict[str, list]:
    """`{slot: terms}`; a plain list applies to the domain's only slot."""
    if isinstance(loss_configs, dict):
        unknown = set(loss_configs) - set(loss_dims)
        if unknown:
            raise KeyError(
                f"LOSS_CONFIGS names slots {sorted(unknown)} the domain lacks"
            )
        return dict(loss_configs)
    if len(loss_dims) != 1:
        raise ValueError(
            f"the domain has slots {sorted(loss_dims)}; give LOSS_CONFIGS per slot"
        )
    return {next(iter(loss_dims)): list(loss_configs)}


@OBJECTIVE.register("reconstruction")
class ReconstructionObjective:
    """Weighted sensecraft terms of `(x[slot], x_hat[slot])`, summed over slots.

    Term logs keep their sensecraft names for a single slot and become `<slot>/<term>` when
    several slots are compared. The loss networks are frozen.
    """

    def __init__(self, loss_configs: list | dict, loss_dims: dict[str, int]):
        terms = per_slot(loss_configs, loss_dims)
        self.prefix = len(terms) > 1
        self.loss = nn.ModuleDict(
            {
                slot: Loss(list(configs), dims=loss_dims[slot])
                .requires_grad_(False)
                .eval()
                for slot, configs in terms.items()
            }
        )

    def compile(self, **kwargs) -> None:
        """Compile every term that holds a network; parameter-free terms stay eager."""
        for loss in self.loss.values():
            for part in loss.parts.values():
                if any(True for _ in part.parameters()):
                    part.compile(**kwargs)

    def to(
        self, device, weights: torch.dtype | None = None
    ) -> "ReconstructionObjective":
        """Move the terms; `weights` holds the frozen feature nets in that dtype."""
        for loss in self.loss.values():
            loss.prepare(device, weights)
        return self

    def _name(self, slot: str, term: str) -> str:
        return f"{slot}/{term}" if self.prefix else term

    def __call__(self, x: dict, x_hat: dict) -> tuple[Tensor, dict]:
        """`(nll, term logs)` of the reconstruction `x_hat` of `x`."""
        total, logs = None, {}
        for slot, loss in self.loss.items():
            out = loss(x_hat[slot], x[slot])
            logs.update({self._name(slot, k): v.detach() for k, v in out.terms.items()})
            total = out.total if total is None else total + out.total
        return total, {**logs, "loss": total.detach()}

    def seeded(self, x: dict, x_hat: dict, scale: float) -> tuple[dict, Tensor, dict]:
        """`(d nll / d x_hat per slot, nll, term logs)`, no graph kept through the nets.

        A slot no loss reads gets a zero seed.
        """
        seeds, total, logs = {}, None, {}
        for slot, value in x_hat.items():
            if slot not in self.loss:
                seeds[slot] = torch.zeros_like(value)
                continue
            out = self.loss[slot].value_and_seed(value, x[slot], scale)
            seeds[slot] = out.seed / scale
            logs.update({self._name(slot, k): v for k, v in out.terms.items()})
            total = out.total if total is None else total + out.total
        return seeds, total, {**logs, "loss": total}
