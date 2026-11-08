"""Topology-aware reductions: on one process every call returns its argument."""

import torch
import torch.distributed as dist


def active() -> bool:
    """True when a process group with more than one rank is live."""
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def world_size() -> int:
    """Rank count, 1 when not distributed."""
    return dist.get_world_size() if active() else 1


def rank() -> int:
    """This rank, 0 when not distributed."""
    return dist.get_rank() if active() else 0


def barrier() -> None:
    """Wait for every rank.

    An NCCL barrier names this rank's device, as Lightning's does: left out, NCCL guesses
    cuda:<rank> and binds the communicator there, so `DEVICES=[2, 3]` then fails the first
    gradient all-reduce with a tensor on the wrong device.
    """
    if not active():
        return
    if dist.get_backend() == "nccl":
        dist.barrier(device_ids=[torch.cuda.current_device()])
    else:
        dist.barrier()


def all_sum_(t: torch.Tensor) -> torch.Tensor:
    """In-place sum-all-reduce."""
    if active():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


def all_mean_(t: torch.Tensor) -> torch.Tensor:
    """In-place mean-all-reduce."""
    if active():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t.div_(dist.get_world_size())
    return t


def all_max_(t: torch.Tensor) -> torch.Tensor:
    """In-place max-all-reduce."""
    if active():
        dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t


def broadcast_(t: torch.Tensor, src: int = 0) -> torch.Tensor:
    """In-place broadcast from `src`."""
    if active():
        dist.broadcast(t, src=src)
    return t
