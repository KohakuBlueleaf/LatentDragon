"""Question: which M, N, K reach the card's matmul peak, and where do conv shapes land?

Times `A(M, K) @ B(K, N)` over a grid of shapes in fp16 and bf16 with autocast off, checks the
dtypes, names the kernel that ran for the best shape, and times 3x3 convs of the recipes in both
dtypes. A conv is an implicit gemm with M pixels, N output channels and K input channels times
the kernel area. Run on an idle GPU. Real compute.

    kogine run experiments/kernels/gemm_shapes.py
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

DEVICE: str = "cuda:0"
SIDES: list[int] = [256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
DTYPES: list[str] = ["float16", "bfloat16"]
MIB_LIMIT: int = 6000  # largest matrix, in MiB of 16-bit values
TOP: int = 12
OUT: Path = Path("runs/bench/gemm_shapes.jsonl")


def best_seconds(run, rounds: int = 6) -> float:
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


def gemm_row(m: int, n: int, k: int, dtype: torch.dtype) -> dict:
    """The rate of one shape, with its dtypes checked."""
    a = torch.randn(m, k, device=DEVICE, dtype=dtype)
    b = torch.randn(k, n, device=DEVICE, dtype=dtype)
    assert a.dtype == dtype and b.dtype == dtype
    assert torch.matmul(a[:8], b[:, :8]).dtype == dtype
    seconds = best_seconds(lambda: torch.matmul(a, b))
    return {
        "m": m,
        "n": n,
        "k": k,
        "dtype": str(dtype),
        "t_mac_s": m * n * k / seconds / 1e12,
    }


def kernel_of(m: int, n: int, k: int, dtype: torch.dtype) -> list[str]:
    """Names of the device kernels one matmul of this shape runs."""
    a = torch.randn(m, k, device=DEVICE, dtype=dtype)
    b = torch.randn(k, n, device=DEVICE, dtype=dtype)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as trace:
        torch.matmul(a, b)
        torch.cuda.synchronize()
    return sorted({e.name for e in trace.events() if "sm" in e.name and "_" in e.name})[
        :6
    ]


@torch.no_grad()
def conv_row(batch: int, channels: int, side: int, dtype, layout) -> dict:
    """The rate of a real bias-free 3x3 conv, as its implicit gemm shape."""
    x = torch.randn(batch, channels, side, side, device=DEVICE, dtype=dtype)
    w = torch.randn(channels, channels, 3, 3, device=DEVICE, dtype=dtype)
    x, w = x.to(memory_format=layout), w.to(memory_format=layout)
    seconds = best_seconds(lambda: F.conv2d(x, w, padding=1))
    return {
        "conv_m": batch * side * side,
        "conv_n": channels,
        "conv_k": channels * 9,
        "dtype": str(dtype),
        "layout": str(layout),
        "t_mac_s": batch * side * side * channels * channels * 9 / seconds / 1e12,
    }


def main() -> None:
    """Print the best shapes per dtype, their kernels, then the conv shapes."""
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with OUT.open("a", encoding="utf-8") as file:
        for name in DTYPES:
            dtype = getattr(torch, name)
            found = []
            for m in SIDES:
                for n in SIDES:
                    for k in SIDES:
                        largest = max(m * k, k * n, m * n) * 2 / 2**20
                        if largest > MIB_LIMIT or m * n * k < 2**31:
                            continue
                        try:
                            found.append(gemm_row(m, n, k, dtype))
                        except torch.OutOfMemoryError:
                            torch.cuda.empty_cache()
            found.sort(key=lambda row: -row["t_mac_s"])
            rows.extend(found)
            print(name, len(found), "shapes", flush=True)
            for row in found[:TOP]:
                print(row, flush=True)
            top = found[0]
            print(kernel_of(top["m"], top["n"], top["k"], dtype), flush=True)
            for batch, channels, side in (
                (16, 128, 256),
                (16, 256, 128),
                (16, 512, 64),
            ):
                for layout in (torch.preserve_format, torch.channels_last):
                    row = conv_row(batch, channels, side, dtype, layout)
                    rows.append(row)
                    print(row, flush=True)
        for row in rows:
            file.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
