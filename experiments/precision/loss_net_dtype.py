"""Question: which compute mode keeps a perceptual term's gradient direction when fake is near real?

No training. For LPIPS and DINOv3 ViT, the gradient at `fake` of the term (the trainer's seed,
taken at a loss scale and divided back) is compared with an fp64 reference, for reconstructions
and for real images plus noise of shrinking size. Modes: the feature net in fp16 under autocast
(what the trainer runs), in fp32 with TF32 convs and matmuls, and in strict fp32. Each mode also
gets its forward-plus-seed time on the batch.

    kogine run experiments/precision/loss_net_dtype.py --config <config setting the paths>
"""

import json
import time
from pathlib import Path

import torch
from sensecraft import Loss, Term

from latentdragon_kernels.error_stats import stats

IMAGES: str | None = None  # a folder of images
SIZE: int = 256
BATCH: int = 16
DEVICE: str = "cuda:0"
SCALE: float = 32768.0
TERMS: dict = {
    "lpips": (0.1, {"net": "vgg"}),
    "dino_vit": (100.0, {"model": "base", "use_gram": False}),
}
NOISE: tuple = (1e-2, 1e-3, 1e-4, 1e-5)
MODES: tuple = ("fp16", "fp32_tf32", "fp32")
TIMING_REPEATS: int = 10
OUT: Path = Path("runs/precision/loss_net_dtype.json")


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


def build(name: str, dtype: torch.dtype) -> Loss:
    """One term, its feature net held in `dtype`."""
    weight, kwargs = TERMS[name]
    loss = Loss([Term(name, weight, **kwargs)])
    return loss.prepare(DEVICE, dtype, torch.channels_last).eval()


def seed(loss: Loss, fake, real, mode: str) -> torch.Tensor:
    """The unscaled seed in `mode`."""
    if mode == "fp16":
        with torch.autocast("cuda", dtype=torch.float16):
            return loss.value_and_seed(fake, real, SCALE).seed / SCALE
    tf32 = mode == "fp32_tf32"
    old = torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32
    torch.backends.cudnn.allow_tf32 = torch.backends.cuda.matmul.allow_tf32 = tf32
    try:
        return loss.value_and_seed(fake, real, SCALE).seed / SCALE
    finally:
        torch.backends.cudnn.allow_tf32, torch.backends.cuda.matmul.allow_tf32 = old


def timed(loss: Loss, fake, real, mode: str) -> float:
    """Milliseconds per forward plus seed, after a warmup call."""
    seed(loss, fake, real, mode)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(TIMING_REPEATS):
        seed(loss, fake, real, mode)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / TIMING_REPEATS * 1000


def main() -> None:
    torch.manual_seed(0)
    real = load_images(BATCH).to(DEVICE)
    cases = {f"near_{n:g}": real + n * torch.randn_like(real) for n in NOISE}
    results: dict = {}

    for name in TERMS:
        reference_loss = build(name, torch.float64)
        losses = {
            "fp16": build(name, torch.float16),
            "fp32_tf32": build(name, torch.float32),
            "fp32": build(name, torch.float32),
        }
        for case, fake in cases.items():
            reference = reference_loss.value_and_seed(fake.double(), real.double()).seed
            for mode in MODES:
                row = stats(seed(losses[mode], fake, real, mode), reference)
                results.setdefault(name, {}).setdefault(case, {})[mode] = row
        for mode in MODES:
            fake = cases["near_0.01"]
            results[name].setdefault("ms", {})[mode] = timed(
                losses[mode], fake, real, mode
            )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    for name, rows in results.items():
        print(f"== {name}  ms per call: " + ", ".join(
            f"{m} {v:.1f}" for m, v in rows["ms"].items()
        ))
        for case in cases:
            print(
                f"  {case:12s} "
                + "  ".join(
                    f"{m}: cos {rows[case][m]['cos']:.5f} rel_mean {rows[case][m]['rel_mean']:.3g}"
                    for m in MODES
                )
            )


if __name__ == "__main__":
    main()
