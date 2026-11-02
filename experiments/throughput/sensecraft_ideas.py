"""Question: which sensecraft changes would pay, measured before any change lands there?

Times LPIPS with channels-last pretrained weights, and DINOv3 with its position embeddings
held across calls, against the plain loss, compiled, at one batch. Real compute.

    kogine run experiments/throughput/sensecraft_ideas.py
"""

import json
import os
import time
from pathlib import Path

STAND_IN_HUB = Path("runs/bench_hub").resolve()
os.environ["HF_HOME"] = str(STAND_IN_HUB)
os.environ["HF_HUB_OFFLINE"] = "1"

import torch  # noqa: E402
from sensecraft.loss import (  # noqa: E402
    DinoV3LossConfig,
    LPIPSConfig,
    SenseCraftLoss,
)

DEVICE: str = "cuda:0"
BATCH: int = 32
SIZE: int = 256
ROUNDS: int = 40
OUT: Path = Path("runs/bench/sensecraft_ideas.jsonl")
DTYPE = torch.bfloat16
LPIPS = LPIPSConfig(net="vgg", weight=0.5)
DINO = DinoV3LossConfig(model_type="base", use_gram=False, weight=250)


def loss_of(config, layout=torch.preserve_format) -> SenseCraftLoss:
    """A frozen loss in the compute dtype, with conv weights in `layout`."""
    loss = SenseCraftLoss([config], mode="2d", return_dict=True)
    loss = loss.requires_grad_(False).eval().to(DEVICE, memory_format=layout)
    for param in loss.parameters():
        param.data = param.data.to(DTYPE)
    return loss


def best_ms(loss, x, fake, layout=torch.preserve_format) -> float:
    """Best milliseconds of the loss and its gradient at `fake`."""
    x, fake = x.to(memory_format=layout), fake.to(memory_format=layout)
    best = float("inf")
    for index in range(ROUNDS):
        torch.cuda.synchronize()
        began = time.perf_counter()
        leaf = fake.detach().requires_grad_()
        with torch.autocast("cuda", dtype=DTYPE):
            value = loss(x, leaf, return_dict=True)["loss"]
        torch.autograd.grad(value, leaf)
        torch.cuda.synchronize()
        if index >= 8:
            best = min(best, time.perf_counter() - began)
    return best * 1e3


def hold_positions(module) -> None:
    """Keep the first position embeddings of `module` for every later call."""
    positions = module.model.rope_embeddings
    compute = positions.forward
    held = {}

    def fetch(pixel_values):
        key = pixel_values.shape[-2:]
        if key not in held:
            held[key] = compute(pixel_values)
        return held[key]

    positions.forward = fetch


def main() -> None:
    """Append one row per arm to `OUT`."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    x = torch.rand(BATCH, 3, SIZE, SIZE, device=DEVICE) * 2 - 1
    fake = (x + 0.1 * torch.randn_like(x)).clamp(-1, 1)
    last = torch.channels_last
    rows = []

    for name, config, layout in (
        ("lpips plain", LPIPS, torch.preserve_format),
        ("lpips channels-last", LPIPS, last),
    ):
        for compiled in (False, True):
            loss = loss_of(config, layout)
            if compiled:
                next(iter(loss.loss_modules.values())).compile()
            ms = best_ms(loss, x, fake, layout)
            rows.append({"arm": name, "compiled": compiled, "ms": ms})

    for name, held in (("dinov3 plain", False), ("dinov3 held positions", True)):
        for compiled in (False, True):
            loss = loss_of(DINO)
            module = next(iter(loss.loss_modules.values()))
            if held:
                hold_positions(module)
            if compiled:
                module.compile()
            rows.append(
                {"arm": name, "compiled": compiled, "ms": best_ms(loss, x, fake)}
            )

    with OUT.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
