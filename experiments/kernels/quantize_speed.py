"""Question: is the Triton `quantize` faster than eager at a recipe's shape?

Times forward and backward of both backends with peak memory above the inputs. Real compute.

    kogine run experiments/kernels/quantize_speed.py
"""

import time

import torch

from latentdragon_kernels.registry import resolve

DEVICE: str = "cuda:0"
ROWS: int = 8192
SHAPES: list[list[int]] = [
    [1, 65536, 256],
    [16, 65536, 16],
    [8, 4096, 32],
]  # parts, codes, dim


def timed(quantize, z, codebook) -> dict[str, float]:
    """Best forward and backward milliseconds, and peak MiB over the inputs."""
    best_forward, best_backward = float("inf"), float("inf")
    z, codebook = z.clone().requires_grad_(), codebook.clone().requires_grad_()
    held = torch.cuda.memory_allocated(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    for _ in range(8):
        z.grad, codebook.grad = None, None
        torch.cuda.synchronize()
        began = time.perf_counter()
        z_q, z_q_diff, _, _ = quantize(z, codebook)
        torch.cuda.synchronize()
        middle = time.perf_counter()
        (z_q.sum() + z_q_diff.sum()).backward()
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
    """Print one line per backend and shape."""
    torch.cuda.set_device(DEVICE)
    for parts, codes, dim in SHAPES:
        z = torch.randn(ROWS, parts, dim, device=DEVICE)
        codebook = torch.randn(parts, codes, dim, device=DEVICE)
        for backend in ("eager", "triton"):
            try:
                quantize = resolve("quantize", DEVICE, backend)
                row = timed(quantize, z, codebook)
            except torch.OutOfMemoryError as err:
                row = {"error": str(err)[:80]}
                torch.cuda.empty_cache()
            print(parts, codes, dim, backend, row, flush=True)


if __name__ == "__main__":
    main()
