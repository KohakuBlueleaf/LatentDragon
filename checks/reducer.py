"""Check: two ranks through each reducer give the gradients of one rank on the whole batch.

Run after changing the reducer, the bundle or the backward order. Uses gloo on the CPU.

    kogine run checks/reducer.py
"""

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from latentdragon.training.parallel.reducer import REDUCER
from latentdragon_common.registry import build

PORT: int = 29517
SPECS: list = ["flat", {"name": "bucketed", "bucket_mib": 0.001}]


def net_of() -> nn.Sequential:
    """The same small network on every rank."""
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(8, 64), nn.ReLU(), nn.Linear(64, 8))


def loss_of(net: nn.Module, batch: torch.Tensor) -> torch.Tensor:
    return net(batch).square().mean()


def rank_main(rank: int, world: int, spec, expected: list) -> None:
    """Reduce this rank's half of the batch and compare with `expected`."""
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{PORT}", rank=rank, world_size=world
    )
    torch.set_num_threads(1)
    net = net_of()
    params = list(net.parameters())
    reducer = build(spec, REDUCER, params=params)

    # Two steps, each with one non-final backward and one final backward
    for _ in range(2):
        half = batch_of().chunk(world)[rank]
        reducer.arm(False)
        (0.5 * loss_of(net, half)).backward()
        reducer.arm(True)
        (0.5 * loss_of(net, half)).backward()
        reducer.finish()
        for param, grad in zip(params, expected):
            assert torch.allclose(param.grad, grad, rtol=1e-5, atol=1e-7), spec
        net.zero_grad()
    dist.destroy_process_group()


def batch_of() -> torch.Tensor:
    return torch.randn(8, 8, generator=torch.Generator().manual_seed(1))


def main() -> None:
    """One process per reducer spec, the reference from a plain backward."""
    net = net_of()
    loss_of(net, batch_of()).backward()
    expected = [param.grad.clone() for param in net.parameters()]
    for spec in SPECS:
        mp.spawn(rank_main, args=(2, spec, expected), nprocs=2)
        print(f"ok {spec}")


if __name__ == "__main__":
    main()
