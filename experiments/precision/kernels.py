"""Kernel precision against fp64, in fp16 and bf16, forward and backward.

Question: how far do the Triton kernels sit from fp64, next to plain PyTorch at the same dtype?

Run: kogine run experiments/precision/kernels.py --set DEVICE=cuda:0
Writes one JSON row per (op, shape, dtype, act, tensor) to OUTPUT. Needs a GPU.
"""

import json
from pathlib import Path

import torch
from torch import Tensor

from latentdragon_kernels import resolve
from latentdragon_kernels.error_stats import compare
from latentdragon_kernels.tuning import tuner

DEVICE = "cuda"
OUTPUT = "runs/precision/kernels.jsonl"
SEED = 0
TUNE = "off"
DTYPES = ("float16", "bfloat16")
ACTIVATIONS = ("silu", "gelu", "mish")
# (rows, parts, codes, dim)
QUANTIZE_SHAPES = ((4096, 4, 1024, 16), (65536, 8, 4096, 32), (1024, 1, 65536, 256))
# (groups, rows, in_dim, hidden, out_dim)
MLP_SHAPES = (
    (4, 1024, 16, 256, 48),
    (1, 4096, 64, 512, 256),
    (1, 65536, 16, 2048, 256),
)
MLP_BACKEND = "gemm"


def _leaf(x: Tensor, dtype: torch.dtype | None = None) -> Tensor:
    """A detached copy of `x` (optionally cast) that requires grad."""
    return x.detach().to(dtype or x.dtype).requires_grad_()


def _run_quantize(fn, z, codebook, cotangents, dtype):
    """Outputs and input gradients of one quantize call, plus the indices."""
    z, codebook = _leaf(z), _leaf(codebook)
    with torch.autocast(z.device.type, dtype=dtype, enabled=dtype is not None):
        z_q, z_q_diff, indices, min_dists = fn(z, codebook)
    grads = torch.autograd.grad(
        (z_q, z_q_diff), (z, codebook), [cot.to(z.dtype) for cot in cotangents]
    )
    tensors = {
        "z_q": z_q,
        "z_q_diff": z_q_diff,
        "min_dists": min_dists,
        "grad_z": grads[0],
        "grad_codebook": grads[1],
    }
    return tensors, indices


def _run_mlp(fn, inputs, cotangent, act, dtype):
    """Output and input gradients of one projection call."""
    leaves = [_leaf(x) for x in inputs]
    with torch.autocast(leaves[0].device.type, dtype=dtype, enabled=dtype is not None):
        out = fn(*leaves, act=act)
    grads = torch.autograd.grad(out, leaves, cotangent.to(out.dtype))
    names = ("grad_base", "grad_w1", "grad_b1", "grad_w2", "grad_b2")
    return {"out": out, **dict(zip(names, grads))}


def _rows(meta: dict, kernel: dict, eager: dict, ref: dict) -> list[dict]:
    """One row per tensor of an op call."""
    rows = []
    for name, out in kernel.items():
        rows.append({**meta, "tensor": name, **compare(out, eager[name], ref[name])})
    return rows


def _quantize_rows(shape: tuple, name: str, device: str) -> list[dict]:
    """Rows for quantize at one shape and dtype: kernel and eager against fp64."""
    rows, parts, codes, dim = shape
    dtype = getattr(torch, name)

    # Random inputs and output cotangents
    z = torch.randn(rows, parts, dim, device=device, dtype=torch.float32)
    codebook = torch.randn(parts, codes, dim, device=device, dtype=torch.float32)
    cotangents = [torch.randn_like(z), torch.randn_like(z)]
    typed = z.to(dtype)

    # Triton kernel, eager at the test dtype, eager in fp64 as the reference
    kernel, kernel_idx = _run_quantize(
        resolve("quantize", device, backend="triton"),
        typed,
        codebook,
        cotangents,
        dtype,
    )
    eager_fn = resolve("quantize", device, backend="eager")
    eager, eager_idx = _run_quantize(eager_fn, typed, codebook, cotangents, dtype)
    ref, ref_idx = _run_quantize(
        eager_fn, typed.double(), codebook.double(), cotangents, None
    )

    # Index agreement with the reference, then the per-tensor error rows
    meta = {"op": "quantize", "shape": shape, "dtype": name, "act": None}
    agreement = {
        **meta,
        "tensor": "indices",
        "kernel_agree": (kernel_idx == ref_idx).float().mean().item(),
        "eager_agree": (eager_idx == ref_idx).float().mean().item(),
    }
    return [agreement, *_rows(meta, kernel, eager, ref)]


def _mlp_rows(shape: tuple, name: str, act: str, device: str) -> list[dict]:
    """Rows for mlp_project at one shape, dtype and activation."""
    groups, rows, in_dim, hidden, out_dim = shape
    dtype = getattr(torch, name)

    # Random inputs scaled by 1/sqrt(in_dim) and an output cotangent
    sizes = (
        (groups, rows, in_dim),
        (groups, in_dim, hidden),
        (groups, 1, hidden),
        (groups, hidden, out_dim),
        (groups, 1, out_dim),
    )
    inputs = [
        torch.randn(size, device=device, dtype=torch.float32) / in_dim**0.5
        for size in sizes
    ]
    cotangent = torch.randn(groups, rows, out_dim, device=device, dtype=torch.float32)

    # The kernel, eager at the test dtype, eager in fp64 as the reference
    kernel_fn = resolve("mlp_project", device, backend=MLP_BACKEND)
    eager_fn = resolve("mlp_project", device, backend="eager")
    typed = [x.to(dtype) for x in inputs]
    kernel = _run_mlp(kernel_fn, typed, cotangent, act, dtype)
    eager = _run_mlp(eager_fn, typed, cotangent, act, dtype)
    ref = _run_mlp(eager_fn, [x.double() for x in typed], cotangent, act, None)
    meta = {"op": "mlp_project", "shape": shape, "dtype": name, "act": act}
    return _rows(meta, kernel, eager, ref)


def main() -> None:
    """Write the error rows of every op, shape, dtype and activation to `OUTPUT`."""
    torch.manual_seed(SEED)
    tuner.set_mode(TUNE)
    path = Path(OUTPUT)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for name in DTYPES:
            for shape in QUANTIZE_SHAPES:
                for row in _quantize_rows(shape, name, DEVICE):
                    file.write(json.dumps(row) + "\n")
            for shape in MLP_SHAPES:
                for act in ACTIVATIONS:
                    for row in _mlp_rows(shape, name, act, DEVICE):
                        file.write(json.dumps(row) + "\n")
