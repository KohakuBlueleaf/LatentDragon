"""Question: how fast and how large is each `mlp_project` backend on a 65536-code, 256-channel codebook?

Times forward and backward of every backend at that shape, across the gemm chunk
sizes, with peak memory above the inputs. Real compute.

    kogine run experiments/kernels/mlp_speed.py
"""

import time

import torch

from latentdragon_kernels.backends.gemm import mlp as gemm
from latentdragon_kernels.registry import resolve

DEVICE: str = "cuda:0"
GROUPS: int = 1
ROWS: int = 65536
IN_DIM: int = 16
HIDDEN: int = 2048
OUT_DIM: int = 256
CHUNKS: list[int] = [128, 256, 512, 1024, 2048]


def timed(project, tensors) -> dict[str, float]:
    """Best forward and backward milliseconds, and peak MiB over the inputs."""
    best_forward, best_backward = float("inf"), float("inf")
    leaves = [t.clone().requires_grad_() for t in tensors]
    held = torch.cuda.memory_allocated(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    for _ in range(8):
        for leaf in leaves:
            leaf.grad = None
        torch.cuda.synchronize()
        began = time.perf_counter()
        out = project(*leaves)
        torch.cuda.synchronize()
        middle = time.perf_counter()
        out.backward(torch.ones_like(out) * 1e-3)
        torch.cuda.synchronize()
        best_forward = min(best_forward, middle - began)
        best_backward = min(best_backward, time.perf_counter() - middle)
    peak = torch.cuda.max_memory_allocated(DEVICE) - held
    return {
        "forward_ms": best_forward * 1e3,
        "backward_ms": best_backward * 1e3,
        "peak_mib": peak / 2**20,
    }


def main() -> None:
    """Print one line per backend and chunk size."""
    tensors = [
        torch.randn(GROUPS, ROWS, IN_DIM, device=DEVICE),
        torch.randn(GROUPS, IN_DIM, HIDDEN, device=DEVICE) / IN_DIM**0.5,
        torch.zeros(GROUPS, 1, HIDDEN, device=DEVICE),
        torch.randn(GROUPS, HIDDEN, OUT_DIM, device=DEVICE) / HIDDEN**0.5,
        torch.zeros(GROUPS, 1, OUT_DIM, device=DEVICE),
    ]
    for backend in ("eager", "triton"):
        print(
            backend, timed(resolve("mlp_project", DEVICE, backend), tensors), flush=True
        )
    for chunk in CHUNKS:
        gemm.CHUNK = chunk
        print("gemm", chunk, timed(gemm.mlp_project, tensors), flush=True)


if __name__ == "__main__":
    main()
