"""Train the keyframe + motion toy, then store a packed latent and decode it back.

    python examples/packed_latent/run.py

After training, one batch is encoded, packed (the keyframe tensor and the motion indices),
written to `packed.safetensors` in the run folder, read back and decoded; the decode of the
stored latent must equal the decode of the live one.
"""

from pathlib import Path

import parts  # noqa: F401  registers the key/motion encoder and decoder
import torch
from kohakuengine import load_config_file
from safetensors.torch import load_file, save_file
from sensecraft import Term

from latentdragon.io.export import load_export, save_export
from latentdragon.model.core import tree
from latentdragon.training.checkpoint import latest
from latentdragon.training.checkpoint.partial import checkpoint_weights
from latentdragon.training.entry import run
from latentdragon_common.knobs import supplied_knobs

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "train" / "ae.py"
_KEY, _MOTION = 4, 8

NAME = "packed_latent"
RUNS_DIR = str(_ROOT / "runs" / "examples")
DEVICES = ["cpu"]
PRECISION = "fp32"
KERNELS = "eager"
NUM_WORKERS = 0
TRAIN_SOURCES = [{"name": "synthetic", "shape": (parts.FRAMES, 3, 16, 16), "count": 64}]
ARCH = {
    "domain": "video",
    "encoder": {
        "name": "key_motion",
        "key_channels": 2 * _KEY,
        "motion_channels": _MOTION,
    },
    "latent": {
        "name": "slots",
        "defs": {
            "key": {"name": "kl", "input": "h_key", "output": "key"},
            "motion": {
                "name": "vq",
                "dim": _MOTION,
                "quantizer": {"name": "kohaku", "n_embed": 64},
                "input": "h_motion",
                "output": "motion",
            },
        },
    },
    "decoder": {"name": "key_motion", "key_channels": _KEY, "motion_channels": _MOTION},
    "head": {"name": "conv", "in_channels": 32, "kernel_size": 1, "video": True},
}
LOSS_CONFIGS = [Term("mse")]
AUX_WEIGHTS = {"key/kl": 1e-6, "motion/vq": 1.0}
BATCH_SIZE = 8
MAX_STEPS = 20
FLUSH_EVERY = 5
SAMPLE_EVERY = 10
CKPT_EVERY = 20
EMA_DECAY = 0.0


def stored_round_trip(run_dir: Path) -> float:
    """Encode a batch, store its packed latent, read it back; the decode difference."""
    folder = run_dir / "export"
    save_export(folder, ARCH, checkpoint_weights(latest(run_dir / "checkpoints")))
    model = load_export(folder).eval()

    batch = {"x": torch.rand(4, parts.FRAMES, 3, 16, 16) * 2 - 1}
    with torch.no_grad():
        encoded = model.encode(batch)
        packed = model.latent.pack(encoded.latent)
        path = run_dir / "packed.safetensors"
        tensors = tree.named_tensors(packed)
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(path))

        stored = load_file(str(path))
        latent = {"key": stored["key"], "motion": {"indices": stored["motion.indices"]}}
        live = model.head(model.decode(encoded.latent))["x"]
        read = model.head(model.decode(latent))["x"]
    return (live - read).abs().max().item()


def main() -> float:
    """Train with the train script's defaults under this module's knobs, then round-trip."""
    defaults = supplied_knobs(load_config_file(_SCRIPT).globals_dict)
    run({**defaults, **supplied_knobs(globals())}, str(_SCRIPT))
    newest = sorted((Path(RUNS_DIR) / NAME).iterdir())[-1]
    difference = stored_round_trip(newest)
    print(f"stored latent decodes like the live one: max abs difference {difference}")
    return difference


if __name__ == "__main__":
    main()
