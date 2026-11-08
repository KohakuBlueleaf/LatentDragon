"""Question: where does the LPIPS seed go non-finite under fp16 loss scaling?

No training. Reconstructions come from a saved checkpoint's EMA weights, run once under fp16
autocast; LPIPS is then measured on `(fake, real)` pairs the way the trainer calls it: VGG held
in fp16, channels-last, under fp16 autocast, the seed taken by `value_and_seed` at a loss scale.
Synthetic stress pairs (flat borders, out-of-range pixels, near-identical images) run next to the
reconstructions.

Per pair set it reports, per VGG tap: the feature magnitude, the positions whose channel vector
is all zero or tiny (where `x / (|x| + eps)` has a 1/|x| gradient), and the largest tap gradient
in fp16; then whether the seed is finite at each loss scale. A precision row compares the fp16
seed with plain-PyTorch fp16 (unfused conv, ReLU and pool) against an fp64 reference.

    kogine run experiments/precision/lpips_fp16.py --config <config setting the paths>
"""

import json
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from sensecraft import Loss, Term
from sensecraft.nets import base as net_base
from sensecraft.nets import vgg as net_vgg

from latentdragon.model.core.build import build_autoencoder
from latentdragon_kernels.error_stats import compare

RUN: Path | None = None  # a run folder with config.json and checkpoints/
STEP: int = 10000
IMAGES: str | None = None  # a folder of images
SIZE: int = 256
BATCH: int = 16
BATCHES: int = 16
SCALES: tuple = (1.0, 1024.0, 32768.0)
TERM: str = "lpips"
TERM_KWARGS: dict = {"net": "vgg"}
WEIGHT: float = 0.1  # the term's weight in the recipe
DEVICE: str = "cuda:0"
COMPILED: bool = True  # compile the LPIPS term as the trainer does with COMPILE_LOSS
PRECISION_SCALE: float = 32768.0  # fp16 seeds underflow at scale 1, so compare at this one
OUT: Path = Path("runs/precision/lpips_fp16.json")
TINY = 1e-3  # a tap position whose channel norm is below this has a gradient past 1/TINY


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
    """The run's autoencoder with the EMA weights of checkpoint `STEP`, in eval mode."""
    config = json.loads((RUN / "config.json").read_text())
    knobs = config.get("knobs", config)
    model = build_autoencoder(knobs["ARCH"])
    ema = load_file(RUN / "checkpoints" / f"step_{STEP:08d}" / "ema.safetensors")
    unexpected = model.load_state_dict(ema, strict=False).unexpected_keys
    if unexpected:
        raise ValueError(f"unexpected keys {unexpected[:5]}")
    model.bind_kernels(torch.device(DEVICE))
    return model.to(DEVICE).eval()


@torch.no_grad()
def reconstruct(model, real: torch.Tensor) -> torch.Tensor:
    """Fp32 reconstructions of `real` under fp16 autocast, as training renders them."""
    with torch.autocast("cuda", dtype=torch.float16):
        out, _ = model({"x": real})
    return out["x"].float()


def pair_sets(model, real: torch.Tensor) -> dict[str, torch.Tensor]:
    """Fakes for one real batch: the reconstruction and synthetic stress cases."""
    recon = reconstruct(model, real)
    border = real.clone()
    border[:, :, : SIZE // 5] = -1  # letterbox black band, in both images
    border_recon = recon.clone()
    border_recon[:, :, : SIZE // 5] = -1 + 0.01 * torch.randn_like(
        border_recon[:, :, : SIZE // 5]
    )
    half = real.clone()
    half[..., SIZE // 2 :] += 0.01 * torch.randn_like(half[..., SIZE // 2 :])
    return {
        "recon": (recon, real),
        "near_1e-2": (real + 1e-2 * torch.randn_like(real), real),
        "near_1e-3": (real + 1e-3 * torch.randn_like(real), real),
        "near_1e-4": (real + 1e-4 * torch.randn_like(real), real),
        "near_1e-5": (real + 1e-5 * torch.randn_like(real), real),
        "exact": (real.clone(), real),
        "half_exact": (half, real),
    }


def training_loss() -> Loss:
    """LPIPS as the trainer holds it: fp16 weights, channels-last, compiled per term."""
    loss = Loss([Term(TERM, WEIGHT, **TERM_KWARGS)])
    loss.prepare(DEVICE, torch.float16, torch.channels_last).eval()
    if COMPILED:
        loss.parts[TERM].compile(mode="default", dynamic=False)
    return loss


def plain_forwards(enabled: bool) -> None:
    """Swap the fused conv-ReLU and the index-free pool for plain PyTorch ops, or back."""
    if enabled:
        net_base.ConvReLU.forward = lambda self, x: F.relu(
            F.conv2d(x, self.weight.to(x.dtype), self.bias.to(x.dtype), 1, 1)
        )
        net_vgg.MaxPool2x2.forward = lambda self, x: F.max_pool2d(x, 2, 2)
    else:
        net_base.ConvReLU.forward = PLAIN_RESTORE["conv"]
        net_vgg.MaxPool2x2.forward = PLAIN_RESTORE["pool"]


PLAIN_RESTORE = {
    "conv": net_base.ConvReLU.forward,
    "pool": net_vgg.MaxPool2x2.forward,
}


def tap_report(loss: Loss, fake: torch.Tensor, real: torch.Tensor, scale: float) -> dict:
    """Per tap: feature absmax, zero and tiny channel norms, the fp16 tap gradient absmax."""
    lpips = loss.parts["lpips"]
    found = []

    def taps(x):
        features = lpips.backbone(
            ((x - lpips.shift) / lpips.scale)
            .to(lpips.backbone.dtype)
            .contiguous(memory_format=torch.channels_last)
        )
        for feature in features:
            if feature.requires_grad:
                feature.retain_grad()
        found.append(features)
        return features

    leaf = fake.detach().requires_grad_()
    with torch.autocast("cuda", dtype=torch.float16):
        fake_taps = taps(leaf)
        with torch.no_grad():
            real_taps = taps(real)
        value = lpips.distance(fake_taps, real_taps)
    (value * WEIGHT * scale).backward()

    rows = []
    for index, feature in enumerate(fake_taps):
        norm = feature.float().square().sum(1).sqrt()
        grad = feature.grad
        rows.append(
            {
                "tap": index,
                "feature_absmax": feature.abs().amax().item(),
                "zero_norm_positions": int((norm == 0).sum()),
                "tiny_norm_positions": int((norm < TINY).sum()),
                "min_norm": norm.amin().item(),
                "grad_finite": bool(torch.isfinite(grad).all()),
                "grad_absmax": grad.float().nan_to_num(0, 0, 0).abs().amax().item(),
            }
        )
    return {
        "taps": rows,
        "input_grad_finite": bool(torch.isfinite(leaf.grad).all()),
    }


def plain_loss() -> Loss:
    """The same LPIPS, eager, for the plain-PyTorch baseline."""
    loss = Loss([Term(TERM, WEIGHT, **TERM_KWARGS)])
    return loss.prepare(DEVICE, torch.float16, torch.channels_last).eval()


def layer_report(loss: Loss, fake: torch.Tensor) -> list[float]:
    """Largest magnitude of the residual stream after each ViT layer, in the compute dtype."""
    net = loss.parts[TERM].net
    peaks = []
    hooks = [
        layer.register_forward_hook(
            lambda _, args, out: peaks.append(out.float().abs().amax().item())
        )
        for layer in net.layer
    ]
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            net.forward((((fake + 1) / 2 - loss.parts[TERM].mean) / loss.parts[TERM].std).to(net.dtype))
    finally:
        for hook in hooks:
            hook.remove()
    return peaks


def seeds(loss: Loss, fake, real, scale: float) -> torch.Tensor:
    """The trainer's seed: gradient of `scale * weight * lpips` at `fake`, under autocast."""
    with torch.autocast("cuda", dtype=torch.float16):
        return loss.value_and_seed(fake, real, scale).seed


def reference_seed(fake, real) -> torch.Tensor:
    """The seed at scale 1 in fp64, plain ops, no autocast."""
    loss = Loss([Term(TERM, WEIGHT, **TERM_KWARGS)]).to(DEVICE).double().eval()
    plain_forwards(True)
    try:
        return loss.value_and_seed(fake.double(), real.double()).seed
    finally:
        plain_forwards(False)


def main() -> None:
    torch.manual_seed(0)
    model = load_model()
    loss = training_loss()
    reals = load_images(BATCH * BATCHES)
    results: dict[str, dict] = {}

    for batch_index in range(BATCHES):
        real = reals[batch_index * BATCH : (batch_index + 1) * BATCH].to(DEVICE)
        for name, (fake, target) in pair_sets(model, real).items():
            entry = results.setdefault(
                name, {"batches": 0, "taps": None, "seed_nonfinite": {}, "rows": []}
            )
            entry["batches"] += 1

            if TERM != "lpips":
                peaks = layer_report(loss, fake)
                kept = entry.setdefault("layer_absmax", [0.0] * len(peaks))
                entry["layer_absmax"] = [max(a, b) for a, b in zip(kept, peaks)]
                report = None
            else:
                # Taps at the largest scale: magnitudes, dead positions, fp16 tap gradients
                report = tap_report(loss, fake, target, max(SCALES))
            if report is None:
                pass
            elif entry["taps"] is None:
                entry["taps"] = [dict(row) for row in report["taps"]]
            else:
                for kept, row in zip(entry["taps"], report["taps"], strict=True):
                    kept["feature_absmax"] = max(
                        kept["feature_absmax"], row["feature_absmax"]
                    )
                    kept["zero_norm_positions"] += row["zero_norm_positions"]
                    kept["tiny_norm_positions"] += row["tiny_norm_positions"]
                    kept["min_norm"] = min(kept["min_norm"], row["min_norm"])
                    kept["grad_finite"] &= row["grad_finite"]
                    kept["grad_absmax"] = max(kept["grad_absmax"], row["grad_absmax"])

            # The trainer's seed at each scale
            for scale in SCALES:
                seed = seeds(loss, fake, target, scale)
                bad = int((~torch.isfinite(seed)).sum())
                key = f"{scale:g}"
                entry["seed_nonfinite"][key] = entry["seed_nonfinite"].get(key, 0) + bad
                peak = seed.nan_to_num(0, 0, 0).abs().amax().item() / scale
                kept = entry.setdefault("seed_absmax_unscaled", {})
                kept[key] = max(kept.get(key, 0.0), peak)

            # Precision of the unscaled seed on the first batch only, taken at a training scale
            if batch_index == 0:
                scale = PRECISION_SCALE
                candidate = seeds(loss, fake, target, scale) / scale
                plain_forwards(True)
                try:
                    baseline = seeds(plain_loss(), fake, target, scale) / scale
                finally:
                    plain_forwards(False)
                reference = reference_seed(fake, target)
                entry["precision"] = compare(candidate, baseline, reference)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))
    for name, entry in results.items():
        print(f"== {name} ({entry['batches']} batches)")
        print("  seed non-finite elements per scale:", entry["seed_nonfinite"])
        print("  seed absmax / scale:", entry["seed_absmax_unscaled"])
        for row in entry["taps"] or []:
            print("  ", row)
        if "layer_absmax" in entry:
            print("  layer absmax:", [round(v, 1) for v in entry["layer_absmax"]])
        if "precision" in entry:
            row = entry["precision"]
            print(
                "  precision mark:",
                row["mark"],
                "cand cos %.6f rel_mean %.3g" % (
                    row["candidate"]["cos"], row["candidate"]["rel_mean"]
                ),
                "base cos %.6f rel_mean %.3g" % (
                    row["baseline"]["cos"], row["baseline"]["rel_mean"]
                ),
            )


if __name__ == "__main__":
    main()
