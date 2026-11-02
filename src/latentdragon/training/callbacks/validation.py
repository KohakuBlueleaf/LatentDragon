"""Validation, sample logging and checkpoint callbacks."""

from pathlib import Path

import torch

from latentdragon.domain.latents import pca_rgb
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
        self.metric_log.record(state.step, {**metrics, "epoch": state.epoch})


class SampleCallback(Callback):
    """Writes a sample grid of a held batch to `samples/<step>.webp` and publishes it.

    Rows: the real images, the latent before quantization as RGB, the reconstruction, then one
    reconstruction per nested latent level when the bottleneck has them.
    """

    KEY = "train/samples"

    def __init__(self, every, task, batch, out_dir, autocast, publish):
        self.every = every
        self.task = task
        self.batch = batch
        self.out_dir = Path(out_dir)
        self.autocast = autocast
        self.publish = publish

    @torch.no_grad()
    def on_step_end(self, state) -> None:
        """Reconstruct the held batch, save the grid and send it to the image sinks."""
        self.task.eval()
        with self.autocast():
            recon, out = self.task.reconstruct(self.batch)
            levels = self.task.model.bottleneck.eval_masks()
            full = out.z.shape[1]
            nested = [
                self.task.reconstruct(self.batch, keep)[0]
                for keep in levels
                if keep < full
            ]
        self.task.train()

        latents = out.raw if out.raw is not None else out.z
        fakes = [fake.float() for fake in (recon, *nested)]
        domain = self.task.model.domain
        grid = domain.to_grid(self.batch.float(), fakes, pca_rgb(latents))

        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"{state.step}.webp"
        domain.save_grid(grid, path)
        self.publish(state.step, self.KEY, path)


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
