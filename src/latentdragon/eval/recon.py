"""Validation metrics: rFID, PSNR and SSIM, plus total code usage."""

import copy

import torch
import torch.distributed as dist
from sensecraft.metrics import psnr, ssim
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance

from latentdragon.eval.usage import CodeTally, NoTally


def new_fid(device) -> FrechetInceptionDistance:
    """An FID metric keeping its real features across resets.

    The first build downloads the Inception weights, so rank 0 builds first and the other
    ranks wait, then load the cached file instead of downloading it at the same time.
    """
    distributed = dist.is_available() and dist.is_initialized()
    first = not distributed or dist.get_rank() == 0
    if not first:
        dist.barrier()
    metric = FrechetInceptionDistance(reset_real_features=False, normalize=True)
    if distributed and first:
        dist.barrier()
    return metric.to(device).eval().requires_grad_(False)


def share_fid(first: FrechetInceptionDistance) -> FrechetInceptionDistance:
    """A second metric reusing `first`'s Inception and real statistics."""
    return copy.deepcopy(first, {id(first.inception): first.inception})


def _psnr(recon: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return psnr(recon, real).mean()


def _ssim(recon: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
    return ssim(recon, real, 1.0).mean()


FRAME_METRICS = {"psnr": _psnr, "ssim": _ssim}


class ReconEvaluator:
    """Holds a rank's validation signals; `run(task)` returns `val/*` metrics."""

    def __init__(self, dataset, batch_size, workers, device, precision, sampler=None):
        self.device = device
        self.device_type = torch.device(device).type
        self.precision = precision
        self.batch_size = batch_size
        self.loader = DataLoader(
            dataset, batch_size=batch_size, num_workers=workers, sampler=sampler
        )
        self.signals: torch.Tensor | None = None
        self.fids: dict[int | None, FrechetInceptionDistance] = {}
        self.use_fid = True

    def _prepare(self, frames, keeps: list) -> None:
        """Read the reference once; one Inception feeds every FID."""
        batches = [batch for batch in self.loader]
        self.signals = torch.cat(batches, dim=0)
        if not self.use_fid:
            return
        first = new_fid(self.device)
        for batch in batches:
            real = frames(batch.to(self.device)).float() * 0.5 + 0.5
            with torch.autocast(self.device_type, enabled=False):
                first.update(real, True)
        self.fids = {keep: share_fid(first) for keep in keeps[1:]}
        self.fids[keeps[0]] = first

    @torch.no_grad()
    def run(self, task, sync) -> dict:
        """Evaluate every nested channel count, or the whole latent."""
        # Metric names and the channel counts to evaluate
        domain = task.model.domain
        frames = domain.to_frames
        self.use_fid = "rfid" in domain.metrics
        names = [name for name in domain.metrics if name in FRAME_METRICS]
        masks = task.model.bottleneck.eval_masks()
        keeps = list(masks) if masks else [None]
        if self.signals is None:
            self._prepare(frames, keeps)

        # Accumulators
        totals = {keep: torch.zeros(len(names), device=self.device) for keep in keeps}
        size = getattr(task.model.bottleneck, "codebook_size", 0)
        tally = CodeTally(task.model.bottleneck, self.device) if size else NoTally()
        seen = torch.zeros((), device=self.device)
        for fid in self.fids.values():
            fid.reset()

        # Reconstruct every batch at every channel count and accumulate metrics
        for start in range(0, self.signals.shape[0], self.batch_size):
            batch = self.signals[start : start + self.batch_size].to(self.device)
            seen += batch.shape[0]
            real = frames(batch.float()) * 0.5 + 0.5
            for keep in keeps:
                with self.precision.autocast(self.device_type):
                    recon, out = task.reconstruct(batch, keep)
                recon = frames(recon.float().clamp(-1, 1)) * 0.5 + 0.5
                with torch.autocast(self.device_type, enabled=False):
                    if self.use_fid:
                        self.fids[keep].update(recon, False)
                    totals[keep] += torch.stack(
                        [FRAME_METRICS[n](recon, real) * batch.shape[0] for n in names]
                    )
            tally.add(out.indices)
        sync.all_sum_(seen)
        return self._metrics(names, keeps, totals, seen, tally, sync)

    def _metrics(self, names, keeps, totals, seen, tally, sync) -> dict:
        """Reduce totals across ranks into `val/*` metrics, with a `-keep` suffix."""
        metrics = {}
        for keep in keeps:
            suffix = "" if keep is None else f"-{keep}"
            sync.all_sum_(totals[keep])
            values = totals[keep] / seen
            if self.use_fid:
                metrics[f"val/rfid{suffix}"] = self.fids[keep].compute().item()
            for index, name in enumerate(names):
                metrics[f"val/{name}{suffix}"] = values[index].item()
        metrics.update(tally.report(sync))
        return metrics
