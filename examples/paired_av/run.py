"""Train the paired video and audio toy for a few CPU steps.

    python examples/paired_av/run.py

Losses are per data slot; the KL of the unified latent is the `kl` latent loss.
"""

from pathlib import Path

import parts  # noqa: F401  registers the toy source, domain, encoder, decoder and head
from kohakuengine import load_config_file
from sensecraft import Term

from latentdragon.training.entry import run
from latentdragon_common.knobs import supplied_knobs

_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _ROOT / "scripts" / "train" / "ae.py"
_LATENT = 4

NAME = "paired_av"
RUNS_DIR = str(_ROOT / "runs" / "examples")
DEVICES = ["cpu"]
PRECISION = "fp32"
KERNELS = "eager"
NUM_WORKERS = 0
TRAIN_SOURCES = [{"name": "toy_av", "count": 64}]
ARCH = {
    "domain": "toy_av",
    "encoder": {"name": "toy_av", "out_channels": 2 * _LATENT},
    "latent": "kl",
    "decoder": {"name": "toy_av", "in_channels": _LATENT},
    "head": "toy_av",
}
LOSS_CONFIGS = {"video": [Term("mse")], "audio": [Term("l1")]}
AUX_WEIGHTS = {"kl": 1e-6}
BATCH_SIZE = 8
MAX_STEPS = 20
FLUSH_EVERY = 5
SAMPLE_EVERY = 0
CKPT_EVERY = 20


def main():
    """Train with the train script's defaults under this module's knobs."""
    defaults = supplied_knobs(load_config_file(_SCRIPT).globals_dict)
    run({**defaults, **supplied_knobs(globals())}, str(_SCRIPT))


if __name__ == "__main__":
    main()
