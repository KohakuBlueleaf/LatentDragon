"""Question: how far are a model's gradients from fp64 under each precision and backend?

Runs one forward and backward of a recipe's model on the same weights and batch, with
fp64 as the reference, plain PyTorch at the same dtype as the baseline, and the kernels or
compiled setup as the candidate. Writes one row per model part: max, mean and median of the
relative and absolute error, cosine, and a triage mark. Real compute.

    kogine run experiments/precision/model_grads.py --set RECIPE=recipes_kvq_f16c256
"""

import dataclasses
import json
from pathlib import Path

import torch
from kohakuengine import load_config_file
from PIL import Image
from torchvision.transforms.functional import center_crop, pil_to_tensor, resize

from latentdragon.io.legacy import read_lightning
from latentdragon.training.build.assemble import build_task
from latentdragon_common.knobs import supplied_knobs
from latentdragon_kernels.error_stats import compare

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "train" / "ae.py"

RECIPE: str = "recipes_kvq_f16c256"
DEVICE: str = "cuda:0"
BATCH: int = 4
SIZE: int = 256
SEED: int = 0
DTYPES: list[str] = ["float16", "bfloat16"]
CANDIDATE: dict = {"KERNELS": "auto", "COMPILE": "off"}
CHECKPOINT: str | None = None  # trained legacy weights; random init when None
IMAGES: str | None = None  # a folder of real images; noise when None
FP16_SCALE: float = 2.0**8  # the loss scale of the fp16 gradients
OUT: Path = Path("runs/precision/model_grads.jsonl")


def knobs_of(**changes) -> dict:
    """Script defaults under the recipe, with `changes` on top."""
    values = supplied_knobs(dict(load_config_file(SCRIPT).globals_dict))
    for folder in ("configs/ae/recipes", "configs/ae/bench"):
        path = ROOT / folder / f"{RECIPE}.py"
        if path.exists():
            values.update(supplied_knobs(dict(load_config_file(path).globals_dict)))
    values.pop("FRAMES", None)
    return {
        **values,
        "EPOCHS": 0,
        "MAX_STEPS": 1,
        "GRAD_CKPT": False,
        "COMPILE": "off",
        "LOSS_16BIT": False,
        **changes,
    }


def part_of(name: str) -> str:
    """The group a parameter reports under: the part, and the stage for the trunk."""
    pieces = name.split(".")
    if pieces[0] == "trunk" and pieces[1] == "stages":
        return f"trunk.{pieces[2]}"
    return pieces[0]


def batch_of() -> torch.Tensor:
    """Real images from `IMAGES`, else noise, in [-1, 1]."""
    if IMAGES is None:
        return torch.rand(BATCH, 3, SIZE, SIZE, device=DEVICE) * 2 - 1
    paths = sorted(Path(IMAGES).rglob("*.jpg"))[: BATCH * 50 : 50]
    frames = []
    for path in paths:
        with Image.open(path) as image:
            pixels = pil_to_tensor(image.convert("RGB"))
        frames.append(center_crop(resize(pixels, SIZE), SIZE))
    return torch.stack(frames).to(DEVICE).float() / 127.5 - 1


def gradients(
    task, x: torch.Tensor, dtype: torch.dtype | None
) -> dict[str, torch.Tensor]:
    """The reconstruction gradient of every parameter, grouped by `part_of`."""
    model = task.model
    model.zero_grad()

    # Eval mode: fp64 and fp32 draw different noise, which would swamp the rounding
    model.eval()
    with torch.autocast("cuda", dtype=dtype, enabled=dtype is not None):
        out = model.bottleneck(model.encode(x))
        fake = model.head(model.decode(out.z))
        nll, _ = task.objective(x, fake)
        loss = nll + sum(out.aux.values(), torch.zeros((), device=x.device))
    scale = FP16_SCALE if dtype == torch.float16 else 1.0
    (loss * scale).backward()

    groups: dict[str, list] = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            groups.setdefault(part_of(name), []).append(param.grad.flatten() / scale)
    return {key: torch.cat(values).double() for key, values in groups.items()}


def main() -> None:
    """Write the rows of each dtype to `OUT`."""
    torch.manual_seed(SEED)
    torch.cuda.set_device(DEVICE)
    torch.backends.cudnn.allow_tf32 = False
    x = batch_of()

    # Trained weights fix the arch through the overrides
    trained = {}
    if CHECKPOINT is not None:
        arch, live, _ = read_lightning(CHECKPOINT)
        trained = {"ARCH_OVERRIDES": dataclasses.asdict(arch)}

    # Reference: eager kernels and everything in fp64
    reference = build_task(knobs_of(KERNELS="eager", **trained), DEVICE)
    if CHECKPOINT is not None:
        reference.model.load_state_dict(live)
    weights = {k: v.clone() for k, v in reference.model.state_dict().items()}
    reference.model.double()
    reference.objective.loss.double()
    ref = gradients(reference, x.double(), None)
    del reference
    torch.cuda.empty_cache()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("a", encoding="utf-8") as file:
        for name in DTYPES:
            dtype = getattr(torch, name)
            baseline = build_task(knobs_of(KERNELS="eager", **trained), DEVICE)
            baseline.model.load_state_dict(weights)
            plain = gradients(baseline, x, dtype)
            del baseline

            candidate = build_task(knobs_of(**{**CANDIDATE, **trained}), DEVICE)
            candidate.model.load_state_dict(weights)
            tried = gradients(candidate, x, dtype)
            del candidate

            for part, grad in ref.items():
                table = compare(tried[part], plain[part], grad)
                row = {"recipe": RECIPE, "dtype": name, "part": part, **table}
                file.write(json.dumps(row) + "\n")
                print(
                    f"{name:9s} {part:12s} {table['mark'] or '-':12s} "
                    f"cand rel {table['candidate']['rel_median']:.2e} "
                    f"base rel {table['baseline']['rel_median']:.2e} "
                    f"cos {table['candidate']['cos']:.6f} "
                    f"{table['baseline']['cos']:.6f}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
