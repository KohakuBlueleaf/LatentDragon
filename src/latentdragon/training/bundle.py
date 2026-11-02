"""`Bundle`: one network's optimizer, schedule, scaler, clip and reducer."""

import torch
from torch import Tensor


def clip_tensors(grads: list[Tensor], limit: float) -> Tensor:
    """The fp32 global norm of `grads`, scaled down to `limit` when `limit` is positive."""
    norms = torch._foreach_norm(grads, 2.0, dtype=torch.float32)
    total = torch.linalg.vector_norm(torch.stack(norms))
    if limit > 0:
        torch._foreach_mul_(grads, (limit / (total + 1e-6)).clamp(max=1.0))
    return total


class Bundle:
    """Owns the update of one parameter group; the driver calls `step` once per step."""

    def __init__(self, name, optimizer, schedule, scaler, reducer, clip: float = 1.0):
        self.name = name
        self.optimizer = optimizer
        self.schedule = schedule
        self.scaler = scaler
        self.reducer = reducer
        self.clip = clip
        self.device_skip = bool(optimizer.defaults.get("fused"))

    def params(self) -> list[torch.nn.Parameter]:
        """Every parameter this bundle updates."""
        return [p for g in self.optimizer.param_groups for p in g["params"]]

    def backward(
        self, scalar: Tensor, retain_graph: bool = False, final: bool = False
    ) -> None:
        """Backward through this bundle's own scaler; `final` marks the step's last one."""
        self.reducer.arm(final)
        self.scaler.scale(scalar).backward(retain_graph=retain_graph)

    def _apply(self, bad: Tensor) -> None:
        """Step unless `bad`: on the device for a fused optimizer, else with a host read."""
        if self.device_skip:
            self.optimizer.found_inf = bad
            self.optimizer.step()
        elif not bad.item():
            self.optimizer.step()

    def _learning_rates(self) -> dict:
        """The learning rate of each group, keyed by group name; older checkpoints have none."""
        return {
            f"optim/lr/{self.name}/{group.get('name', f'group{index}')}": group["lr"]
            for index, group in enumerate(self.optimizer.param_groups)
        }

    def step(self) -> dict[str, Tensor]:
        """Reduce, unscale, clip, step or skip, update scale, advance the schedule."""
        # A bundle with no gradient this step only advances its schedule
        grads = self.reducer.finish()
        if not grads:
            self.schedule.step()
            return self._learning_rates()

        # Undo loss scaling, then clip to the norm limit
        self.scaler.unscale(grads)
        norm = clip_tensors(grads, self.clip)

        # The learning rate each group steps with, and the scale this step's loss was under
        logs = self._learning_rates()
        if isinstance(self.scaler.value, Tensor):
            logs[f"optim/loss_scale/{self.name}"] = self.scaler.value.detach().clone()

        # A non-finite norm skips the step and backs the scale off
        bad = (~torch.isfinite(norm)).to(torch.float32)
        self._apply(bad)
        self.scaler.update(bad)
        self.schedule.step()
        self.optimizer.zero_grad(set_to_none=True)

        logs[f"optim/grad_norm/{self.name}"] = norm.detach()
        logs[f"optim/skipped/{self.name}"] = bad
        return logs

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
