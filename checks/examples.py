"""Examples: the toy slot layouts train, and a stored packed latent decodes like the live one."""

import importlib
import sys

from _tiny import ROOT, run_checks


def example(name: str, folder, **knobs):
    """The `run` module of example `name`, its knobs changed, with its own `parts`."""
    path = str(ROOT / "examples" / name)
    sys.path.insert(0, path)
    for module in ("parts", "run"):
        sys.modules.pop(module, None)
    try:
        module = importlib.import_module("run")
    finally:
        sys.path.remove(path)
    for key, value in {"RUNS_DIR": str(folder), **knobs}.items():
        setattr(module, key, value)
    return module


def paired_av_trains(folder) -> None:
    example("paired_av", folder, MAX_STEPS=4, CKPT_EVERY=4).main()


def packed_latent_round_trips(folder) -> None:
    difference = example("packed_latent", folder, MAX_STEPS=4, CKPT_EVERY=4).main()
    assert difference < 1e-6, difference


if __name__ == "__main__":
    run_checks(paired_av_trains, packed_latent_round_trips)
