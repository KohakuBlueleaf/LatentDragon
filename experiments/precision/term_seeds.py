"""Question: how strong, how spiky and how high-frequency is each loss term's gradient at the fake?

No training. Reconstructions come from a checkpoint's EMA weights; each recipe term's seed (its
weighted gradient at the fake, as the trainer takes it: fp16 autocast, loss scale, divided back)
is measured alone. Per term: L2 norm and its ratio to the MSE seed, peak over RMS, the 99.9th
percentile over the median magnitude, the share of power above a quarter of the sampling rate,
and the cosine between terms.

    kogine run experiments/precision/term_seeds.py --config <config setting the paths>
"""

import json
from pathlib import Path

import torch
from sensecraft import Loss, Term
from torch.nn.functional import cosine_similarity

from latentdragon.model.core.build import build_autoencoder

RUN: Path | None = None
STEP: int = 10000
IMAGES: str | None = None
SIZE: int = 256
BATCH: int = 16
BATCHES: int = 4
DEVICE: str = "cuda:0"
SCALE: float = 32768.0
TERMS: dict = {
    "mse": (1.0, {}),
    "lpips": (0.1, {"net": "vgg"}),
    "dino_vit": (250.0, {"model": "base", "use_gram": False}),
}
HIGH: float = 0.25  # cycles per pixel above which power counts as high frequency
OUT: Path = Path("runs/precision/term_seeds.json")


def load_images(count: int) -> torch.Tensor:
    """`count` images from `IMAGES` in [-1, 1], resized and center-cropped to `SIZE`."""
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


def load_model():
    """The run's autoencoder with the EMA weights of checkpoint `STEP`."""
    from safetensors.torch import load_file

    config = json.loads((RUN / "config.json").read_text())
    model = build_autoencoder(config.get("knobs", config)["ARCH"])
    ema = load_file(RUN / "checkpoints" / f"step_{STEP:08d}" / "ema.safetensors")
    model.load_state_dict(ema, strict=False)
    model.bind_kernels(torch.device(DEVICE))
    return model.to(DEVICE).eval()


def seed(loss: Loss, fake, real) -> torch.Tensor:
    """The weighted seed of one term, as the trainer takes it."""
    with torch.autocast("cuda", dtype=torch.float16):
        return loss.value_and_seed(fake, real, SCALE).seed / SCALE


def high_share(grad: torch.Tensor) -> float:
    """Share of the seed's power above `HIGH` cycles per pixel."""
    power = torch.fft.fft2(grad.double()).abs().square()
    fy = torch.fft.fftfreq(grad.shape[-2], device=grad.device).abs()
    fx = torch.fft.fftfreq(grad.shape[-1], device=grad.device).abs()
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)
    return (power[..., radius > HIGH].sum() / power.sum()).item()


def describe(grad: torch.Tensor) -> dict:
    magnitude = grad.double().abs().flatten()
    rms = magnitude.square().mean().sqrt()
    quantiles = torch.quantile(
        magnitude[torch.randperm(len(magnitude), device=grad.device)[:1_000_000]],
        torch.tensor([0.5, 0.999], dtype=torch.float64, device=grad.device),
    )
    return {
        "norm": grad.double().norm().item(),
        "peak_over_rms": (magnitude.max() / rms).item(),
        "p999_over_median": (quantiles[1] / quantiles[0]).item(),
        "high_share": high_share(grad),
    }


def main() -> None:
    torch.manual_seed(0)
    model = load_model()
    losses = {
        name: Loss([Term(name, weight, **kwargs)])
        .prepare(DEVICE, torch.float16, torch.channels_last)
        .eval()
        for name, (weight, kwargs) in TERMS.items()
    }
    reals = load_images(BATCH * BATCHES)
    rows: dict = {}
    for index in range(BATCHES):
        real = reals[index * BATCH : (index + 1) * BATCH].to(DEVICE)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            fake = model({"x": real})[0]["x"].float()
        seeds = {name: seed(loss, fake, real) for name, loss in losses.items()}
        for name, grad in seeds.items():
            row = describe(grad)
            row["norm_over_mse"] = row["norm"] / seeds["mse"].double().norm().item()
            for other, other_grad in seeds.items():
                if other != name:
                    row[f"cos_{other}"] = cosine_similarity(
                        grad.double().flatten(), other_grad.double().flatten(), dim=0
                    ).item()
            for key, value in row.items():
                rows.setdefault(name, {}).setdefault(key, []).append(value)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rows, indent=1))
    for name, stats in rows.items():
        line = "  ".join(
            f"{key} {sum(v) / len(v):.4g}" for key, v in stats.items() if key != "norm"
        )
        print(f"{name:9s} {line}")


if __name__ == "__main__":
    main()
