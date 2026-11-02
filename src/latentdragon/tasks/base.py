"""`Task`: what the engine asks of a training job."""

from collections.abc import Iterable

import torch

from latentdragon.model.core import tree
from latentdragon_common.registry import Registry

TASK = Registry("task")


class Task:
    """Owns modules and losses; never loops, steps optimizers or touches scalers."""

    ema_group = "model"
    memory_format = torch.preserve_format

    def groups(self) -> dict[str, list[dict]]:
        """Optimizer param groups per bundle name, each with `params` and `lr_scale`."""
        raise NotImplementedError

    def micro_step(self, ctx) -> dict:
        """One micro-batch: forwards plus backward through `ctx.backward[name]`."""
        raise NotImplementedError

    def saved_modules(self) -> dict[str, torch.nn.Module]:
        """Name -> module for checkpoints."""
        raise NotImplementedError

    def summary_modules(self) -> dict[str, torch.nn.Module]:
        """Name -> module for the module summary; the checkpointed ones by default."""
        return self.saved_modules()

    def ema_named_params(self) -> dict[str, torch.nn.Parameter]:
        """Parameters tracked by weight EMA."""
        return {}

    def split(self, batch, micro_count: int) -> Iterable:
        """The micro-batches of one step, every tensor split along dim 0."""
        return tree.chunk(batch, micro_count)

    def to_device(self, batch, device):
        """Every tensor on `device`; 4D tensors take the task's memory format."""
        return tree.to_device(batch, device, self.memory_format)

    def reduce_metrics(self, per_micro: list[dict]) -> dict:
        """Mean each key across the micro-batches that produced it; one micro passes through."""
        if len(per_micro) == 1:
            return per_micro[0]
        merged: dict = {}
        for out in per_micro:
            for key, value in out.items():
                merged.setdefault(key, []).append(value)
        return {k: _mean(v) for k, v in merged.items()}

    def post_step(self, step: int, sync) -> None:
        """Once per step after EMA; sync mutated state here."""
        return None

    def train(self) -> None:
        raise NotImplementedError

    def eval(self) -> None:
        raise NotImplementedError


def _mean(values: list):
    """The mean across micro-batches, element by element for vector values."""
    if isinstance(values[0], torch.Tensor):
        return torch.stack([v.detach().float() for v in values]).mean(dim=0)
    return sum(values) / len(values)
