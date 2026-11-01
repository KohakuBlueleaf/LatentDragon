"""`Task`: what the engine asks of a training job."""

from collections.abc import Iterable

import torch

from latentdragon_common.registry import Registry

TASK = Registry("task")


class Task:
    """Owns modules and losses; never loops, steps optimizers or touches scalers."""

    ema_group = "model"

    def groups(self) -> dict[str, list[dict]]:
        """Optimizer param groups per bundle name, each with `params` and `lr_scale`."""
        raise NotImplementedError

    def micro_step(self, ctx) -> dict:
        """One micro-batch: forwards plus backward through `ctx.backward[name]`."""
        raise NotImplementedError

    def saved_modules(self) -> dict[str, torch.nn.Module]:
        """Name -> module for checkpoints."""
        raise NotImplementedError

    def ema_named_params(self) -> dict[str, torch.nn.Parameter]:
        """Parameters tracked by weight EMA."""
        return {}

    def split(self, batch: torch.Tensor, micro_count: int) -> Iterable:
        """The micro-batches of one step."""
        return batch.chunk(micro_count, dim=0)

    def to_device(self, batch: torch.Tensor, device) -> torch.Tensor:
        return batch.to(device, non_blocking=True)

    def reduce_metrics(self, per_micro: list[dict]) -> dict:
        """Mean each key across the micro-batches that produced it."""
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
    if isinstance(values[0], torch.Tensor):
        return torch.stack([v.detach().float() for v in values]).mean()
    return sum(values) / len(values)
