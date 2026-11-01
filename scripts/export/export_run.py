"""Export a run's weights: native folder, research-repo files or a diffusers folder.

kogine run scripts/export/export_run.py --set CHECKPOINT=... --set FORM=native
"""

import json
from pathlib import Path

from latentdragon.io.export import save_diffusers, save_export
from latentdragon.io.legacy import write_lightning, write_safetensors
from latentdragon.training.build.assemble import arch_config
from latentdragon.training.checkpoint.partial import checkpoint_weights

CHECKPOINT = ""  # a checkpoints/step_XXXXXXXX folder
TARGET = "exported"
FORM = "native"  # native | legacy_safetensors | legacy_ckpt | diffusers
USE_EMA = True
LATENTS_MEAN = None
LATENTS_STD = None


def main():
    """Load the checkpoint's weights and write them in the chosen FORM."""
    folder = Path(CHECKPOINT)
    config = json.loads((folder.parents[1] / "config.json").read_text("utf-8"))
    arch = arch_config(config)
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
        case _:
            raise SystemExit(f"unknown FORM {FORM!r}")


if __name__ == "__main__":
    main()
