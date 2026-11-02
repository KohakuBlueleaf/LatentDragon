"""Question: which layers of a block hold its time?

Runs real blocks of an arch file with a device sync around every leaf layer, forward only, in both
layouts, and prints each layer's milliseconds and share. Run on an idle GPU. Real compute.

    kogine run experiments/kernels/block_layers.py
"""

import time
from pathlib import Path

import torch
from kohakuengine import load_config_file

from latentdragon.model import build_autoencoder
from latentdragon.model.nn.blocks import Block

ARCH_FILE: Path = Path("configs/ae/arch/sdvae.py")
DEVICE: str = "cuda:0"
BATCH: int = 16
SIZE: int = 256
BLOCKS: list[str] = [
    "encoder.stages.0.blocks.0",
    "encoder.stages.1.blocks.1",
    "encoder.stages.3.blocks.0",
]
DTYPE = torch.bfloat16
ROUNDS: int = 6


class LayerClock:
    """Accumulates the synced wall time of each leaf layer."""

    def __init__(self, block):
        self.seconds = {}
        self.began = 0.0
        self.handles = []
        for name, module in block.named_modules():
            if not list(module.children()):
                self.handles.append(module.register_forward_pre_hook(self._start))
                self.handles.append(module.register_forward_hook(self._stop(name)))

    def _start(self, module, inputs) -> None:
        torch.cuda.synchronize()
        self.began = time.perf_counter()

    def _stop(self, name: str):
        def stop(module, inputs, output) -> None:
            torch.cuda.synchronize()
            spent = time.perf_counter() - self.began
            self.seconds[name] = min(self.seconds.get(name, spent), spent)

        return stop


def arch_of(path: Path, **changes) -> dict:
    """The `ARCH` of an arch config file, with parts replaced by `changes`."""
    return {**load_config_file(path).globals_dict["ARCH"], **changes}


def inputs_of(model, x: torch.Tensor) -> dict:
    """The input of every block, from one traced forward."""
    seen = {}
    names = {module: name for name, module in model.named_modules()}

    def record(module, inputs):
        seen[names[module]] = inputs

    handles = [
        m.register_forward_pre_hook(record)
        for m in model.modules()
        if isinstance(m, Block)
    ]
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        model.encode({"x": x})
    for handle in handles:
        handle.remove()
    return seen


def report(block, inputs, name: str, layout) -> None:
    """Print the layer times of `block` on its own input."""
    x = inputs[0].to(memory_format=layout)
    with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
        best = float("inf")
        for _ in range(ROUNDS):
            torch.cuda.synchronize()
            began = time.perf_counter()
            block(x, *inputs[1:])
            torch.cuda.synchronize()
            best = min(best, time.perf_counter() - began)
        clock = LayerClock(block)
        for _ in range(ROUNDS):
            block(x, *inputs[1:])
        for handle in clock.handles:
            handle.remove()

    total = sum(clock.seconds.values())
    print(f"{name} {layout} whole {best * 1e3:.2f} ms, layers {total * 1e3:.2f} ms")
    for layer, seconds in clock.seconds.items():
        print(
            f"  {layer:20s} {seconds * 1e3:7.2f} ms {seconds / total:6.1%}", flush=True
        )


def main() -> None:
    """Print each named block in both layouts."""
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.benchmark = True
    model = build_autoencoder(arch_of(ARCH_FILE)).to(DEVICE).eval()
    seen = inputs_of(model, torch.randn(BATCH, 3, SIZE, SIZE, device=DEVICE))
    for layout in (torch.preserve_format, torch.channels_last):
        model.to(memory_format=layout)
        for name in BLOCKS:
            report(model.get_submodule(name), seen[name], name, layout)


if __name__ == "__main__":
    main()
