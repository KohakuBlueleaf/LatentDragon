"""Gradient reducers: where gradients live, and how they are averaged across ranks.

`bucketed` is the default. Gradients are views into flat buckets, and each bucket is
all-reduced as soon as its last gradient lands in the step's final backward, so the
communication overlaps the rest of backward in small transfers.
"""

from functools import partial

import torch
import torch.distributed as dist
from torch import Tensor

from latentdragon.training.parallel import sync
from latentdragon_common.registry import Registry

REDUCER = Registry("reducer")
MIB = 2**20


@REDUCER.register("none")
class NoReducer:
    """Per-parameter gradients, never reduced; the single-process baseline."""

    def __init__(self, params: list, wire: torch.dtype | None = None):
        self.params = params

    def arm(self, final: bool) -> None:
        """Called before every backward."""
        return None

    def finish(self) -> list[Tensor]:
        """This step's gradients."""
        return [p.grad for p in self.params if p.grad is not None]


@REDUCER.register("flat")
class MeanReducer(NoReducer):
    """One flat all-reduce after backward, optionally through a compressed wire dtype."""

    def __init__(self, params: list, wire: torch.dtype | None = None):
        super().__init__(params)
        self.wire = wire

    def finish(self) -> list[Tensor]:
        """Average every gradient across ranks in place through one flat buffer."""
        grads = super().finish()
        if not grads:
            return grads
        flat = torch.cat([g.reshape(-1) for g in grads])
        wire = flat if self.wire is None else flat.to(self.wire)
        sync.all_mean_(wire)
        restored = wire.to(flat.dtype)

        # Scatter the averaged buffer back into each gradient
        offset = 0
        for grad in grads:
            size = grad.numel()
            grad.copy_(restored[offset : offset + size].view_as(grad))
            offset += size
        return grads


def layout_view(flat: Tensor, param: Tensor) -> Tensor:
    """A view of `flat` with the shape and memory order of `param`."""
    order = sorted(range(param.dim()), key=param.stride, reverse=True)
    stored = flat.view([param.shape[dim] for dim in order])
    return stored.permute([order.index(dim) for dim in range(param.dim())])


class Bucket:
    """Same-dtype parameters whose gradients are views into one flat buffer."""

    def __init__(self, params: list, wire: torch.dtype | None):
        self.params = params
        first = params[0]
        size = sum(p.numel() for p in params)
        self.flat = torch.zeros(size, dtype=first.dtype, device=first.device)
        self.wire = wire if wire != first.dtype else None

        offset = 0
        self.views = []
        for param in params:
            count = param.numel()
            self.views.append(layout_view(self.flat[offset : offset + count], param))
            offset += count

        self.seen = [False] * len(params)
        self.pending = len(params)
        self.work = None
        self.sent = None

    def attach(self) -> None:
        """Zero the buffer and make it every parameter's gradient."""
        self.flat.zero_()
        for param, view in zip(self.params, self.views):
            param.grad = view
        self.seen = [False] * len(self.params)
        self.work = None

    def launch(self, average) -> None:
        """Start this bucket's all-reduce."""
        self.sent = self.flat if self.wire is None else self.flat.to(self.wire)
        self.work = dist.all_reduce(self.sent, op=average, async_op=True)

    def land(self) -> None:
        """Wait for the reduction and take its result."""
        self.work.wait()
        if self.wire is not None:
            self.flat.copy_(self.sent)
        self.sent = None

    def drop_unseen(self) -> None:
        """Parameters no backward reached have no gradient, as under plain autograd."""
        for param, seen in zip(self.params, self.seen):
            if not seen:
                param.grad = None


def buckets_of(params: list, limit: int) -> list[list]:
    """Parameters in backward order, one dtype per bucket, cut every `limit` bytes."""
    by_dtype: dict[torch.dtype, list] = {}
    for param in reversed(params):
        by_dtype.setdefault(param.dtype, []).append(param)

    groups = []
    for members in by_dtype.values():
        current, size = [], 0
        for param in members:
            if current and size >= limit:
                groups.append(current)
                current, size = [], 0
            current.append(param)
            size += param.numel() * param.element_size()
        groups.append(current)
    return groups


@REDUCER.register("bucketed")
class BucketedReducer:
    """Flat gradient buckets, each reduced while the step's final backward runs.

    On one process it keeps the buckets, so clipping and norms take a few launches.
    """

    def __init__(
        self, params: list, wire: torch.dtype | None = None, bucket_mib: float = 8.0
    ):
        limit = int(bucket_mib * MIB)
        self.buckets = [Bucket(group, wire) for group in buckets_of(params, limit)]
        self.world = sync.world_size()
        self.premul = self.world > 1 and dist.get_backend() == "nccl"
        self.average = dist.ReduceOp.SUM
        if self.premul:
            self.average = dist._make_nccl_premul_sum(1.0 / self.world)
        self.attached = False
        self.final = False
        self.next = 0
        self.hooks = [
            param.register_post_accumulate_grad_hook(partial(self._arrive, b, i))
            for b, bucket in enumerate(self.buckets)
            for i, param in enumerate(bucket.params)
        ]

    def arm(self, final: bool) -> None:
        """Attach the buckets at the step's first backward; `final` opens launching."""
        if not self.attached:
            for bucket in self.buckets:
                bucket.attach()
            self.attached = True
            self.next = 0
        self.final = final
        if final:
            for bucket in self.buckets:
                bucket.pending = len(bucket.params)

    def _arrive(self, index: int, position: int, param) -> None:
        """A gradient landed; launch every complete bucket, in order on every rank."""
        bucket = self.buckets[index]
        bucket.seen[position] = True
        if not self.final:
            return
        bucket.pending -= 1
        while self.next < len(self.buckets) and self.buckets[self.next].pending == 0:
            self._launch(self.buckets[self.next])
            self.next += 1

    def _launch(self, bucket: Bucket) -> None:
        if self.world > 1:
            bucket.launch(self.average)

    def finish(self) -> list[Tensor]:
        """Launch what is left, wait for every reduction, return the flat gradients."""
        if not self.attached:
            return []
        self.attached = False
        self.final = False

        for bucket in self.buckets[self.next :]:
            self._launch(bucket)
        for bucket in self.buckets:
            if bucket.work is not None:
                bucket.land()
                if not self.premul:
                    bucket.flat.div_(self.world)
            bucket.drop_unseen()
        return [bucket.flat for bucket in self.buckets]
