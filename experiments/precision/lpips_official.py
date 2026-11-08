"""Question: does our LPIPS match the official `lpips` package, and where does its gradient come from?

No training. Reconstructions come from a checkpoint's EMA weights. The official package (0.1.4,
unpacked at `OFFICIAL`, imported from there) and SenseCraft's LPIPS VGG are compared on the same
pairs: the learned head weights, the value and the gradient at the fake at fp64, the official one
at strict fp32, and ours on the trainer's fp16 path. The gradient is then split by tap: each tap
alone, its share of the full gradient's norm and its power above a quarter of the sampling rate.

    kogine run experiments/precision/lpips_official.py --config <config setting the paths>
"""

import json
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file
from sensecraft import Loss, Term
from sensecraft.distance import unit_channels
from sensecraft.weights import packaged
from torch.nn.functional import cosine_similarity

from latentdragon.model.core.build import build_autoencoder

RUN: Path | None = None
STEP: int = 10000
IMAGES: str | None = None
OFFICIAL: str | None = None  # folder holding the unpacked official `lpips` package
SIZE: int = 256
BATCH: int = 8
DEVICE: str = "cuda:0"
SCALE: float = 32768.0
HIGH: float = 0.25
OUT: Path = Path("runs/precision/lpips_official.json")


def load_images(count: int) -> torch.Tensor:
    from PIL import Image
    from torchvision.transforms.functional import center_crop, pil_to_tensor, resize

    suffixes = (".jpg", ".jpeg", ".png", ".webp")
    paths = sorted(p for p in Path(IMAGES).rglob("*") if p.suffix.lower() in suffixes)
    paths = paths[:: max(len(paths) // count, 1)][:count]
    frames = []
    for path in paths:
        with Image.open(path) as image:
            pixels = pil_to_tensor(image.convert("RGB"))
        frames.append(center_crop(resize(pixels, SIZE, antialias=True), SIZE))
    return torch.stack(frames).float() / 127.5 - 1


def reconstructions(real: torch.Tensor) -> torch.Tensor:
    config = json.loads((RUN / "config.json").read_text())
    model = build_autoencoder(config.get("knobs", config)["ARCH"])
    ema = load_file(RUN / "checkpoints" / f"step_{STEP:08d}" / "ema.safetensors")
    model.load_state_dict(ema, strict=False)
    model.bind_kernels(torch.device(DEVICE))
    model = model.to(DEVICE).eval()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
        return model({"x": real})[0]["x"].float()


def grad_of(fn, fake, real) -> tuple[float, torch.Tensor]:
    """Mean value of `fn(fake, real)` and its gradient at `fake`."""
    leaf = fake.detach().requires_grad_()
    value = fn(leaf, real).mean()
    (grad,) = torch.autograd.grad(value, leaf)
    return value.item(), grad


def cos(a, b) -> float:
    return cosine_similarity(a.double().flatten(), b.double().flatten(), dim=0).item()


def high_share(grad: torch.Tensor) -> float:
    power = torch.fft.fft2(grad.double()).abs().square()
    fy = torch.fft.fftfreq(grad.shape[-2], device=grad.device).abs()
    fx = torch.fft.fftfreq(grad.shape[-1], device=grad.device).abs()
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return (power[..., radius > HIGH].sum() / power.sum()).item()


def main() -> None:
    sys.path.insert(0, OFFICIAL)
    import lpips as official_lpips

    torch.manual_seed(0)
    real = load_images(BATCH).to(DEVICE)
    fake = reconstructions(real)
    out: dict = {}

    official = official_lpips.LPIPS(net="vgg", verbose=False).to(DEVICE).eval()

    # Head weights: the official 1x1 lin convs against our packaged heads
    ours_heads = packaged("lpips_vgg")
    out["head_max_abs_diff"] = max(
        (official.lins[i].model[-1].weight.detach().cpu() - ours_heads[f"lins.{i}"])
        .abs()
        .max()
        .item()
        for i in range(5)
    )

    # Value and gradient: official fp64 is the reference
    official64 = official.double()
    ref_value, ref_grad = grad_of(official64, fake.double(), real.double())
    official32 = official_lpips.LPIPS(net="vgg", verbose=False).to(DEVICE).eval()
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    value32, grad32 = grad_of(official32, fake, real)
    torch.backends.cudnn.conv.fp32_precision = "tf32"
    value_tf32, grad_tf32 = grad_of(official32, fake, real)

    ours64 = Loss([Term("lpips", net="vgg")]).to(DEVICE).double().eval()
    ours64_out = ours64.value_and_seed(fake.double(), real.double())
    ours16 = (
        Loss([Term("lpips", net="vgg")])
        .prepare(DEVICE, torch.float16, torch.channels_last)
        .eval()
    )
    with torch.autocast("cuda", dtype=torch.float16):
        ours16_out = ours16.value_and_seed(fake, real, SCALE)
    out["value"] = {
        "official_fp64": ref_value,
        "official_fp32": value32,
        "official_tf32": value_tf32,
        "ours_fp64": ours64_out.total.item(),
        "ours_fp16": ours16_out.total.item(),
    }
    out["grad_cos_to_official_fp64"] = {
        "official_fp32": cos(grad32, ref_grad),
        "official_tf32": cos(grad_tf32, ref_grad),
        "ours_fp64": cos(ours64_out.seed, ref_grad),
        "ours_fp16": cos(ours16_out.seed / SCALE, ref_grad),
    }
    out["grad_norm_ratio_to_official_fp64"] = {
        "ours_fp64": (ours64_out.seed.norm() / ref_grad.norm()).item(),
        "ours_fp16": ((ours16_out.seed / SCALE).double().norm() / ref_grad.norm()).item(),
    }

    # The gradient split by tap, in fp64 with our net
    lpips = ours64.parts["lpips"]
    heads = [h.double() for h in lpips.heads()]
    with torch.no_grad():
        real_taps = lpips.features(real.double())
    full = ours64_out.seed
    taps = []
    for index in range(5):

        def one_tap(x, _real, index=index):
            feature = lpips.features(x)[index]
            diff = (unit_channels(feature) - unit_channels(real_taps[index])).square()
            return (diff * heads[index]).sum(1).mean((1, 2)).mean()

        value, grad = grad_of(one_tap, fake.double(), real.double())
        taps.append(
            {
                "tap": index,
                "value": value,
                "norm_share": (grad.norm() / full.norm()).item(),
                "cos_to_full": cos(grad, full),
                "high_share": high_share(grad),
                "peak_over_rms": (grad.abs().max() / grad.square().mean().sqrt()).item(),
            }
        )
    out["taps"] = taps
    out["full_high_share"] = high_share(full)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "taps"}, indent=1))
    for row in taps:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})


if __name__ == "__main__":
    main()
