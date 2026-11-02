"""Question: where does a perceptual loss spend its time?

Splits LPIPS and DINOv3 into the target forward, the input forward and the backward at one
batch, eager and compiled, in the loss network's compute dtype. Weight values do not matter;
the DINOv3 row uses the bench's random ViT-B. Real compute.

    kogine run experiments/throughput/loss_parts.py
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
ROUNDS: int = 30
OUT: Path = Path("runs/bench/loss_parts.jsonl")
DTYPE = torch.bfloat16
LOSSES = {
    "lpips_vgg": LPIPSConfig(net="vgg", weight=0.5),
    "dinov3_vitb": DinoV3LossConfig(model_type="base", use_gram=False, weight=250),
}


def best_ms(run) -> float:
    """Best milliseconds of `run` after a warm-up."""
    best = float("inf")
    for index in range(ROUNDS):
        torch.cuda.synchronize()
        began = time.perf_counter()
        run()
        torch.cuda.synchronize()
        if index >= 5:
            best = min(best, time.perf_counter() - began)
    return best * 1e3


def features_of(module):
    """The pretrained network's forward on pixels in the module's own input range."""
    if hasattr(module, "dinov3_fwd"):
        return lambda x: module.dinov3_fwd(module.normalize_input((x + 1) / 2))
    return lambda x: module.net(module.scaling_layer(x))


def row_of(name: str, config, compiled: bool, x, fake) -> dict:
    """Part times of one loss."""
    loss = SenseCraftLoss([config], mode="2d", return_dict=True)
    loss = loss.requires_grad_(False).eval().to(DEVICE)
    for param in loss.parameters():
        param.data = param.data.to(DTYPE)
    module = next(iter(loss.loss_modules.values()))
    if compiled:
        module.compile()
    features = features_of(getattr(module, "_orig_mod", module))

    def target_forward() -> None:
        with torch.no_grad(), torch.autocast("cuda", dtype=DTYPE):
            features(x)

    def input_forward() -> None:
        with torch.autocast("cuda", dtype=DTYPE):
            features(fake.detach().requires_grad_())

    def full() -> None:
        leaf = fake.detach().requires_grad_()
        with torch.autocast("cuda", dtype=DTYPE):
            value = loss(x, leaf, return_dict=True)["loss"]
        torch.autograd.grad(value, leaf)

    target, forward, whole = (
        best_ms(target_forward),
        best_ms(input_forward),
        best_ms(full),
    )
    return {
        "loss": name,
        "compiled": compiled,
        "target_forward_ms": target,
        "input_forward_ms": forward,
        "full_ms": whole,
        "backward_ms": whole - target - forward,
    }


def main() -> None:
    """Append one row per loss and arm to `OUT`."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    x = torch.rand(BATCH, 3, SIZE, SIZE, device=DEVICE) * 2 - 1
    fake = (x + 0.1 * torch.randn_like(x)).clamp(-1, 1)
    with OUT.open("a", encoding="utf-8") as file:
        for name, config in LOSSES.items():
            for compiled in (False, True):
                row = row_of(name, config, compiled, x, fake)
                file.write(json.dumps(row) + "\n")
                print(row, flush=True)


if __name__ == "__main__":
    main()
