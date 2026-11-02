"""The training loop: micro-steps, every bundle steps, EMA, `post_step`."""

from dataclasses import dataclass
from typing import Any

import torch

from latentdragon.training.state import TrainState


@dataclass
class MicroCtx:
    """One micro-batch handed to `Task.micro_step`."""

    batch: Any
    micro_index: int
    micro_count: int
    step: int
    backward: dict
    timer: Any
    sync: Any
    seed_scale: float = 1.0


class Driver:
    """Owns step order; a Task never steps an optimizer."""

    def __init__(self, task, bundles, ema, precision, device, micro_count, sync, timer):
        self.task = task
        self.bundles = bundles
        self.ema = ema
        self.precision = precision
        self.device = device
        self.device_type = torch.device(device).type
        self.micro_count = micro_count
        self.sync = sync
        self.timer = timer
        self.backward = {name: b.backward for name, b in bundles.items()}

    def micro_steps(self, batch, step: int) -> dict:
        """Slice the batch into `micro_count` micro-batches and run the task on each."""
        outputs = []
        for micro_index, chunk in enumerate(self.task.split(batch, self.micro_count)):
            ctx = MicroCtx(
                chunk,
                micro_index,
                self.micro_count,
                step,
                self.backward,
                self.timer,
                self.sync,
                self.bundles["model"].scaler.value,
            )
            with self.precision.autocast(self.device_type):
                outputs.append(self.task.micro_step(ctx))
        return self.task.reduce_metrics(outputs)

    def step(self, batch, state: TrainState) -> dict:
        """One optimizer step in the fixed order."""
        with self.timer.phase("micro"):
            metrics = self.micro_steps(batch, state.step)
        with self.timer.phase("bundles"):
            for bundle in self.bundles.values():
                metrics.update(bundle.step())
        with self.timer.phase("ema"):
            self.ema.update()
        with self.timer.phase("post"):
            self.task.post_step(state.step, self.sync)
        return metrics

    def run(self, batches, state: TrainState, steps: int, end_step) -> None:
        """Loop until `steps`; `end_step(state, metrics, batch)` handles logging and cadence."""
        iterator = iter(batches)
        while state.step < steps:
            self.timer.start_step(state.step)
            with self.timer.phase("data"):
                batch, state.epoch, state.batches = next(iterator)
                batch = self.task.to_device(batch, self.device)
            metrics = self.step(batch, state)
            state.step += 1
            end_step(state, metrics, batch)
