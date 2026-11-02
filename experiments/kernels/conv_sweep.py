"""Question: how much conv rate is lost when the feature map no longer fits in cache?

Runs one 3x3 conv, forward and input gradient, at fixed channels across growing tensor sizes,
in both layouts, with cuDNN benchmark on, after the raw bf16 matmul peaks. Prints GMAC/s and
the tensor size. Run on an idle GPU. Real compute.

    kogine run experiments/kernels/conv_sweep.py
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

DEVICE: str = "cuda:0"
BATCHES: list[int] = [1, 2, 4, 8, 16, 32, 64]
SIDE: int = 128
CHANNELS: list[int] = [128, 256, 512]
DTYPE = torch.bfloat16
BYTES = 2
OUT: Path = Path("runs/bench/conv_sweep.jsonl")


def best_seconds(run, rounds: int = 10) -> float:
    """Best wall seconds of `run` after two warm-ups."""
    best = float("inf")
    for index in range(rounds):
        torch.cuda.synchronize()
        began = time.perf_counter()
        run()
        torch.cuda.synchronize()
        if index >= 2:
            best = min(best, time.perf_counter() - began)
    return best


def row_of(channels: int, batch: int, layout) -> dict:
    """Forward and input-gradient rate of one conv."""
    x = torch.randn(batch, channels, SIDE, SIDE, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(channels, channels, 3, 3, device=DEVICE, dtype=DTYPE)
    x, weight = x.to(memory_format=layout), weight.to(memory_format=layout)
    macs = batch * channels * channels * 9 * (SIDE - 2) ** 2

    with torch.no_grad():
        forward = best_seconds(lambda: F.conv2d(x, weight))

    leaf = x.clone().requires_grad_()

    def both():
        torch.autograd.grad(F.conv2d(leaf, weight).sum(), leaf)

    whole = best_seconds(both)
    return {
        "channels": channels,
        "batch": batch,
        "layout": str(layout),
        "tensor_mib": x.numel() * BYTES / 2**20,
        "forward_gmac_s": macs / forward / 1e9,
        "backward_gmac_s": macs / max(whole - forward, 1e-9) / 1e9,
    }


@torch.no_grad()
def gemm_peaks() -> list[dict]:
    """Raw matmul rates in bf16: small batched and one large, the tensor-core ceiling."""
    rows = []
    for batch, side in ((4096, 128), (256, 512), (1, 4096)):
        a = torch.randn(batch, side, side, device=DEVICE, dtype=DTYPE)
        b = torch.randn(batch, side, side, device=DEVICE, dtype=DTYPE)
        seconds = best_seconds(lambda: torch.bmm(a, b))
        rate = batch * side**3 / seconds / 1e9
        rows.append({"gemm_batch": batch, "side": side, "gmac_s": rate})
    return rows


def main() -> None:
    """Print and append the gemm peaks, then one row per channels, batch and layout."""
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as file:
        for row in gemm_peaks():
            file.write(json.dumps(row) + "\n")
            print(row, flush=True)
        for channels in CHANNELS:
            for layout in (torch.preserve_format, torch.channels_last):
                for batch in BATCHES:
                    try:
                        row = row_of(channels, batch, layout)
                    except torch.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        continue
                    file.write(json.dumps(row) + "\n")
                    print(row, flush=True)


if __name__ == "__main__":
    main()
