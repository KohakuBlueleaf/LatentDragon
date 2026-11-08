"""Export a run's weights: native folder, research-repo files, a diffusers or a Hub folder.

kogine run scripts/export/export_run.py --set CHECKPOINT=... --set FORM=native

`FORM=hf` writes a folder that `transformers` and `diffusers` load with `AutoModel` and
`trust_remote_code=True`, and `latentdragon.io.hf.from_pretrained` loads as well; `CARD` is a
markdown file copied in as its `README.md`.
"""

import json
from pathlib import Path

from latentdragon.io.arch_json import upgrade_arch
from latentdragon.io.export import save_diffusers, save_export
from latentdragon.io.hf import save_pretrained
from latentdragon.io.legacy import write_lightning, write_safetensors
from latentdragon.training.checkpoint.partial import checkpoint_weights

CHECKPOINT = ""  # a checkpoints/step_XXXXXXXX folder
TARGET = "exported"
FORM = "native"  # native | legacy_safetensors | legacy_ckpt | diffusers | hf
USE_EMA = True
LATENTS_MEAN = None
LATENTS_STD = None
CARD = None  # a markdown model card for FORM=hf


def main():
    """Load the checkpoint's weights and write them in the chosen FORM."""
    folder = Path(CHECKPOINT)
    config = json.loads((folder.parents[1] / "config.json").read_text("utf-8"))
    arch = upgrade_arch(config["ARCH"])
    state = checkpoint_weights(folder, USE_EMA)
    match FORM:
        case "native":
            save_export(TARGET, arch, state)
        case "legacy_safetensors":
            write_safetensors(TARGET, arch, state)
        case "legacy_ckpt":
            write_lightning(TARGET, arch, state)
        case "diffusers":
            save_diffusers(TARGET, arch, state, LATENTS_MEAN, LATENTS_STD)
        case "hf":
            card = Path(CARD).read_text("utf-8") if CARD else None
            save_pretrained(TARGET, arch, state, card)
        case _:
            raise SystemExit(f"unknown FORM {FORM!r}")


if __name__ == "__main__":
    main()
