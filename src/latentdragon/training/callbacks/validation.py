"""Validation, sample logging and checkpoint callbacks."""

from pathlib import Path

import torch

from latentdragon.training.callbacks.runner import Callback


class ValidationCallback(Callback):
    """Runs the evaluator on EMA weights and logs `val/*`."""

    at_end = True

    def __init__(self, every, evaluator, task, ema, sync, metric_log):
        self.every = every
        self.evaluator = evaluator
        self.task = task
        self.ema = ema
        self.sync = sync
        self.metric_log = metric_log

    def on_step_end(self, state) -> None:
        """Evaluate with EMA weights swapped in, then log the metrics."""
        self.task.eval()
        with self.ema.swapped():
            metrics = self.evaluator.run(self.task, self.sync)
        self.task.train()
        self.metric_log.record(state.step, metrics)


class SampleCallback(Callback):
    """Writes a real-over-recon grid for a held batch to `samples/<step>.webp`."""

    def __init__(self, every, task, batch, out_dir, autocast):
        self.every = every
        self.task = task
        self.batch = batch
        self.out_dir = Path(out_dir)
        self.autocast = autocast

    @torch.no_grad()
    def on_step_end(self, state) -> None:
        """Reconstruct the held batch and save the real-over-recon grid."""
        self.task.eval()
        with self.autocast():
            recon, _ = self.task.reconstruct(self.batch)
        self.task.train()

        grid = self.task.model.domain.to_grid(self.batch.float(), [recon.float()])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.task.model.domain.save_grid(grid, self.out_dir / f"{state.step}.webp")


class CheckpointCallback(Callback):
    """Saves every interval and after the final step."""

    at_end = True

    def __init__(self, every, manager, task, bundles, ema):
        self.every = every
        self.manager = manager
        self.task = task
        self.bundles = bundles
        self.ema = ema

    def on_step_end(self, state) -> None:
        """Save the full training state at this step."""
        self.manager.save(state, self.task.saved_modules(), self.bundles, self.ema)
