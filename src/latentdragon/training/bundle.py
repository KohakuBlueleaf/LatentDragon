"""`Bundle`: one network's optimizer, schedule, scaler, clip and reducer."""

from collections.abc import Callable

import torch
from torch.nn.utils import clip_grad_norm_


def norm_only(params: list, limit: float) -> torch.Tensor:
    """The pre-clip norm with clipping selected off."""
    return clip_grad_norm_(params, float("inf"))


class Bundle:
    """Owns the update of one parameter group; the driver calls `step` once per step."""

    def __init__(self, name, optimizer, schedule, scaler, reducer, clip: float = 1.0):
        self.name = name
        self.optimizer = optimizer
        self.schedule = schedule
        self.scaler = scaler
        self.reducer = reducer
        self.clip = clip
        self._clip: Callable = clip_grad_norm_ if clip > 0 else norm_only

    def params(self) -> list[torch.nn.Parameter]:
        """Every parameter this bundle updates."""
        return [p for g in self.optimizer.param_groups for p in g["params"]]

    def backward(self, scalar: torch.Tensor, retain_graph: bool = False) -> None:
        """Backward through this bundle's own scaler."""
        self.scaler.scale(scalar).backward(retain_graph=retain_graph)

    def step(self) -> dict[str, float]:
        """Reduce, unscale, clip, step or skip, update scale, advance the schedule."""
        # A bundle with no gradient this step only advances its schedule.
        with_grad = [p for p in self.params() if p.grad is not None]
        if not with_grad:
            self.schedule.step()
            return {}

        # Average across ranks, undo loss scaling, then clip to the norm limit.
        grads = [p.grad for p in with_grad]
        self.reducer.reduce(grads)
        self.scaler.unscale(grads)
        norm = self._clip(with_grad, self.clip)

        # A non-finite norm skips the optimizer step and backs the scale off.
        finite = bool(torch.isfinite(norm))
        if finite:
            self.optimizer.step()
        self.scaler.update(finite)
        self.schedule.step()
        self.optimizer.zero_grad(set_to_none=True)

        return {
            f"grad_norm/{self.name}": norm.detach(),
            f"skipped/{self.name}": not finite,
        }

    def state_dict(self) -> dict:
        """Optimizer, schedule and scaler state."""
        return {
            "optimizer": self.optimizer.state_dict(),
            "schedule": self.schedule.state_dict(),
            "scaler": self.scaler.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.schedule.load_state_dict(state["schedule"])
        self.scaler.load_state_dict(state["scaler"])
