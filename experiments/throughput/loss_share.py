"""Question: what does each loss cost, forward and backward, per dtype and compile setup?

Times one loss at a time on a fixed batch, with the seed backward the task really runs, and
prints one JSON row per (loss, arm). Weight values do not matter here; the DINOv3 row uses the
bench's random ViT-B. Real compute.

    kogine run experiments/throughput/loss_share.py
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
    GeneralConfig,
    LPIPSConfig,
    SenseCraftLoss,
)

DEVICE: str = "cuda:0"
BATCH: int = 32
SIZE: int = 256
ROUNDS: int = 30
OUT: Path = Path("runs/bench/loss_share.jsonl")
LOSSES = {
    "mse": GeneralConfig("mse", weight=1.0),
    "lpips_vgg": LPIPSConfig(net="vgg", weight=0.5),
    "dinov3_vitb": DinoV3LossConfig(model_type="base", use_gram=False, weight=250),
}
ARMS = [
    {"dtype": "bfloat16", "compile": False, "weights16": False},
    {"dtype": "bfloat16", "compile": False, "weights16": True},
    {"dtype": "bfloat16", "compile": True, "weights16": False},
    {"dtype": "bfloat16", "compile": True, "weights16": True},
    {"dtype": "float16", "compile": True, "weights16": True},
    {"dtype": "float32", "compile": False, "weights16": False},
]


def timed(loss, x, fake, dtype) -> float:
    """Best milliseconds of one forward plus the gradient at `fake`."""
    best = float("inf")
    for round_index in range(ROUNDS):
        torch.cuda.synchronize()
        begin = time.perf_counter()
        leaf = fake.detach().requires_grad_()
        with torch.autocast("cuda", dtype=dtype):
            value = loss(x, leaf.to(dtype), return_dict=True)["loss"]
        torch.autograd.grad(value, leaf)
        torch.cuda.synchronize()
        if round_index >= 5:
            best = min(best, time.perf_counter() - begin)
    return best * 1e3


def main() -> None:
    """Append one row per loss and arm to `OUT`."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    x = torch.rand(BATCH, 3, SIZE, SIZE, device=DEVICE) * 2 - 1
    fake = (x + 0.1 * torch.randn_like(x)).clamp(-1, 1)
    with OUT.open("a", encoding="utf-8") as file:
        for name, config in LOSSES.items():
            for arm in ARMS:
                loss = SenseCraftLoss([config], mode="2d", return_dict=True)
                loss = loss.requires_grad_(False).eval().to(DEVICE)
                dtype = getattr(torch, arm["dtype"])
                if arm["weights16"]:
                    for param in loss.parameters():
                        param.data = param.data.to(dtype)
                if arm["compile"]:
                    for module in loss.loss_modules.values():
                        if any(True for _ in module.parameters()):
                            module.compile()
                row = {"loss": name, **arm, "ms": timed(loss, x, fake, dtype)}
                file.write(json.dumps(row) + "\n")
                print(row, flush=True)


if __name__ == "__main__":
    main()
