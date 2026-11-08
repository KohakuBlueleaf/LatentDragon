"""Question: which op's fp16 rounding destroys a perceptual term's gradient when fake is near real?

No training. LPIPS and DINOv3 ViT run in fp64; one point at a time is rounded to fp16 and back,
either the value it passes forward or the gradient it passes backward. A point is the image the
net reads, or the output of one module. The seed at `fake` is compared with the unrounded fp64
seed by cosine. A point whose rounding alone drops the cosine is where the mantissa is lost.

    kogine run experiments/precision/mantissa_points.py --config <config setting IMAGES>
"""

import json
from pathlib import Path

import torch
from sensecraft import Loss, Term
from torch.nn.functional import cosine_similarity

IMAGES: str | None = None
SIZE: int = 256
BATCH: int = 8
DEVICE: str = "cuda:0"
NOISE: tuple = (1e-3, 1e-4, 1e-5)
SCALE: float = 32768.0  # the loss scale, so backward rounding sees training-size gradients
TERMS: dict = {
    "lpips": (0.1, {"net": "vgg"}),
    "dino_vit": (100.0, {"model": "base", "use_gram": False}),
}
OUT: Path = Path("runs/precision/mantissa_points.json")


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


def round16(x: torch.Tensor) -> torch.Tensor:
    """`x` rounded to fp16 and back to its own dtype."""
    return x.to(torch.float16).to(x.dtype)


class RoundGrad(torch.autograd.Function):
    """Identity forward; the gradient passing back is rounded to fp16."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return round16(grad)


def rounding(kind: str):
    """A forward hook rounding a module's output (`fwd`) or the gradient it gets (`bwd`)."""

    def hook(module, args, out):
        if not isinstance(out, torch.Tensor):
            return out
        return round16(out) if kind == "fwd" else RoundGrad.apply(out)

    return hook


def points(name: str, loss: Loss) -> dict[str, torch.nn.Module]:
    """The modules whose outputs are rounded, by name, in call order."""
    net = loss.parts[name].backbone if name == "lpips" else loss.parts[name].net
    found = {}
    for path, module in net.named_modules():
        if not path or isinstance(module, torch.nn.Identity):
            continue
        if name == "lpips" and path.count(".") == 1:  # layers.<i>
            found[path] = module
        if name == "dino_vit" and (
            path.startswith("embeddings")
            and path.count(".") == 0
            or path.count(".") == 1
            or path.count(".") == 2
            and path.split(".")[-1] in ("norm1", "attention", "norm2", "mlp")
        ):
            found[path] = module
    return found


def input_point(name: str, loss: Loss, kind: str):
    """Round the image the net reads, by wrapping the term's `features`."""
    part = loss.parts[name]
    original = part.features

    def features(x):
        x = round16(x) if kind == "fwd" else RoundGrad.apply(x)
        return original(x)

    part.features = features
    return lambda: setattr(part, "features", original)


def seed(loss: Loss, fake, real) -> torch.Tensor:
    return loss.value_and_seed(fake, real, SCALE).seed


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def main() -> None:
    torch.manual_seed(0)
    real = load_images(BATCH).to(DEVICE).double()
    results: dict = {}

    for name, (weight, kwargs) in TERMS.items():
        loss = Loss([Term(name, weight, **kwargs)]).to(DEVICE).double().eval()
        modules = points(name, loss)
        for noise in NOISE:
            fake = real + noise * torch.randn_like(real)
            reference = seed(loss, fake, real)
            rows = {}
            for kind in ("fwd", "bwd"):
                undo = input_point(name, loss, kind)
                rows[f"{kind} input"] = cos(seed(loss, fake, real), reference)
                undo()
                for path, module in modules.items():
                    handle = module.register_forward_hook(rounding(kind))
                    rows[f"{kind} {path}"] = cos(seed(loss, fake, real), reference)
                    handle.remove()

                # Every point at once, for the total of this direction
                handles = [m.register_forward_hook(rounding(kind)) for m in modules.values()]
                undo = input_point(name, loss, kind)
                rows[f"{kind} ALL"] = cos(seed(loss, fake, real), reference)
                undo()
                for handle in handles:
                    handle.remove()
            # The fix candidate: input and first layer exact, every later output rounded both ways
            if name == "lpips":
                later = [m for p, m in modules.items() if p != "layers.0"]
            else:
                later = [
                    m
                    for p, m in modules.items()
                    if p.split(".")[-1] in ("norm1", "attention", "norm2", "mlp")
                ]
            handles = [m.register_forward_hook(rounding("fwd")) for m in later]
            handles += [m.register_forward_hook(rounding("bwd")) for m in later]
            rows["FIX: input+first layer exact (+dino stream), rest fp16"] = cos(
                seed(loss, fake, real), reference
            )
            for handle in handles:
                handle.remove()

            # DINO: every sub-block output and the patch embedding rounded both ways, the
            # residual stream and the input kept exact
            if name == "dino_vit":
                inner = [
                    m
                    for p, m in modules.items()
                    if p.split(".")[-1] in ("norm1", "attention", "norm2", "mlp")
                ]
                inner.append(loss.parts[name].net.embeddings.patch_embeddings)
                for kind in ("fwd", "bwd"):
                    handles = [m.register_forward_hook(rounding(kind)) for m in inner]
                    rows[f"{kind} SUBBLOCKS (stream+input exact)"] = cos(
                        seed(loss, fake, real), reference
                    )
                    for handle in handles:
                        handle.remove()
                handles = [m.register_forward_hook(rounding("fwd")) for m in inner]
                handles += [m.register_forward_hook(rounding("bwd")) for m in inner]
                rows["both SUBBLOCKS (stream+input exact)"] = cos(
                    seed(loss, fake, real), reference
                )
                for handle in handles:
                    handle.remove()
            results.setdefault(name, {})[f"{noise:g}"] = rows
            print(f"== {name} noise {noise:g}")
            for key, value in rows.items():
                print(f"  {key:40s} cos {value:.5f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
