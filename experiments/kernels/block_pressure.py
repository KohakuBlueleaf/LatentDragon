"""Question: how far from the card's compute and memory peaks does each block run?

Measures a compute peak (a bare wide conv) and memory peaks (read and copy over a buffer far
past the cache). Then times every block of an arch file alone, forward and backward,
and counts its MACs, the bytes its layers read and write, and the bytes that remain when the
block is fused so only its input and output touch memory. Run on an idle GPU. Real compute.

    kogine run experiments/kernels/block_pressure.py
"""

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from kohakuengine import load_config_file
from torch import nn

from latentdragon.io.arch_json import arch_from_json
from latentdragon.model import AEArchConfig, build_autoencoder
from latentdragon.model.blocks import Block

ARCH_FILE: Path = Path("configs/ae/arch/sdvae.py")
DEVICE: str = "cuda:0"
BATCH: int = 16
SIZE: int = 256
DTYPE = torch.bfloat16
BYTES = 2
L2_MIB: float = 88.0  # RTX 5090 L2, from the card's public spec
OUT: Path = Path("runs/bench/block_pressure.jsonl")


def arch_of(path: Path, **changes) -> AEArchConfig:
    """The `ARCH` of an arch config file, with fields replaced by `changes`."""
    return arch_from_json({**load_config_file(path).globals_dict["ARCH"], **changes})


def best_seconds(run, rounds: int = 12) -> float:
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


@torch.no_grad()
def peaks() -> dict[str, float]:
    """Measured peaks: bare conv, GMAC/s; read and copy past the cache, GB/s."""
    channels, side = 512, 128
    x = torch.randn(16, channels, side, side, device=DEVICE, dtype=DTYPE)
    weight = torch.randn(channels, channels, 3, 3, device=DEVICE, dtype=DTYPE)
    x = x.to(memory_format=torch.channels_last)
    weight = weight.to(memory_format=torch.channels_last)
    macs = 16 * channels * channels * 9 * (side - 2) ** 2
    conv = best_seconds(lambda: F.conv2d(x, weight))

    big = torch.ones(2**30, device=DEVICE, dtype=torch.float32)
    other = torch.zeros_like(big)
    size = big.numel() * 4
    read = best_seconds(lambda: big.sum())
    copy = best_seconds(lambda: other.copy_(big))
    return {
        "conv_gmac_s": macs / conv / 1e9,
        "conv_input_mib": x.numel() * BYTES / 2**20,
        "read_gb_s": size / read / 1e9,
        "copy_gb_s": 2 * size / copy / 1e9,
    }


class Counter:
    """Counts one block's work through hooks on its leaf layers."""

    def __init__(self):
        self.macs = 0
        self.elements = 0
        self.moved = 0
        self.widest = 0

    def leaf(self, module, inputs, output) -> None:
        """Add one leaf layer's arithmetic and traffic."""
        tensors = [t for t in inputs if isinstance(t, torch.Tensor)] + [output]
        self.moved += sum(t.numel() * BYTES for t in tensors)
        self.moved += sum(p.numel() * BYTES for p in module.parameters(recurse=False))
        self.widest = max(self.widest, *(t.numel() * BYTES for t in tensors))
        if isinstance(module, nn.Conv2d):
            per_output = module.in_channels // module.groups
            area = module.kernel_size[0] * module.kernel_size[1]
            self.macs += output.numel() * per_output * area
        else:
            self.elements += output.numel()


def leaves(module: nn.Module):
    """Leaf layers of `module`, the blocks' own parts."""
    return [m for m in module.modules() if not list(m.children())]


def measure(block: Block, x: torch.Tensor, args: tuple, label: str) -> dict:
    """One block alone: timed forward and backward, then one counted forward."""
    params = [p for p in block.parameters()]
    x = x.detach().requires_grad_()

    def forward():
        with torch.autocast("cuda", dtype=DTYPE):
            return block(x, *args)

    forward_s = best_seconds(forward)

    def both():
        (forward().float().square().mean()).backward()
        block.zero_grad()

    both_s = best_seconds(both)

    count = Counter()
    handles = [m.register_forward_hook(count.leaf) for m in leaves(block)]
    out = forward()
    for handle in handles:
        handle.remove()
    fused = (x.numel() + out.numel()) * BYTES
    fused += sum(p.numel() * BYTES for p in params)
    return {
        "block": label,
        "forward_ms": forward_s * 1e3,
        "backward_ms": (both_s - forward_s) * 1e3,
        "gmacs": count.macs / 1e9,
        "moved_gb": count.moved / 1e9,
        "fused_gb": fused / 1e9,
        "widest_mib": count.widest / 2**20,
        "input_mib": x.numel() * BYTES / 2**20,
    }


def block_inputs(model, x) -> list:
    """Every block with its real input, from one traced forward."""
    seen = []

    def record(module, inputs):
        seen.append((module, inputs))

    handles = [
        m.register_forward_pre_hook(record)
        for m in model.modules()
        if isinstance(m, Block)
    ]
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        out = model.bottleneck(model.encode(x))
        model.head(model.decode(out.z))
    for handle in handles:
        handle.remove()
    return seen


def main() -> None:
    """Print peaks, then one row per block with its compute and memory MFU."""
    torch.cuda.set_device(DEVICE)
    top = peaks()
    print(top, flush=True)

    model = build_autoencoder(arch_of(ARCH_FILE)).to(DEVICE).train()
    x = torch.randn(BATCH, 3, SIZE, SIZE, device=DEVICE)
    names = {module: name for name, module in model.named_modules()}
    rows = []
    for module, inputs in block_inputs(model, x):
        row = measure(module, inputs[0], inputs[1:], names[module])
        forward_s = row["forward_ms"] / 1e3
        row["gmac_s"] = row["gmacs"] / forward_s
        row["compute_mfu"] = row["gmac_s"] / top["conv_gmac_s"]
        row["memory_mfu"] = row["moved_gb"] / forward_s / top["copy_gb_s"]
        row["fused_share"] = row["fused_gb"] / row["moved_gb"]
        row["past_l2"] = row["widest_mib"] > L2_MIB
        rows.append(row)
        print(
            {k: round(v, 3) if isinstance(v, float) else v for k, v in row.items()},
            flush=True,
        )

    # Whole-model forward totals
    total_s = sum(r["forward_ms"] for r in rows) / 1e3
    total = {
        "forward_ms": total_s * 1e3,
        "backward_ms": sum(r["backward_ms"] for r in rows),
        "gmacs": sum(r["gmacs"] for r in rows),
        "moved_gb": sum(r["moved_gb"] for r in rows),
        "fused_gb": sum(r["fused_gb"] for r in rows),
    }
    total["compute_mfu"] = total["gmacs"] / total_s / top["conv_gmac_s"]
    total["memory_mfu"] = total["moved_gb"] / total_s / top["copy_gb_s"]
    print({k: round(v, 3) for k, v in total.items()}, flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as file:
        for row in [{"peaks": top}, *rows, {"total": total}]:
            file.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
