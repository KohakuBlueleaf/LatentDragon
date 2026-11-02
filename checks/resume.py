"""Resume and checkpoint behavior: an interrupted run continues exactly."""

import torch
from _tiny import SCRIPT, TINY_CRITIC, knobs, run_checks, run_dir_of, train
from safetensors.torch import load_file

from latentdragon.training.callbacks.validation import CheckpointCallback
from latentdragon.training.checkpoint import latest
from latentdragon.training.entry import run

OPTIONS = dict(
    MAX_STEPS=4,
    EMA_DECAY=0.9,
    GRAD_ACC=2,
    GAN_SCHEME={"name": "taming", "disc_start": 0},
    CRITIC=TINY_CRITIC,
)


def saved(run_dir, name: str) -> dict:
    """One tensor file of the run's newest checkpoint."""
    return load_file(str(latest(run_dir / "checkpoints") / name))


def assert_identical(first: dict, second: dict) -> None:
    assert set(first) == set(second)
    for key, value in first.items():
        assert torch.equal(value, second[key]), key


def crash_after_first_save(values: dict) -> None:
    """Train until the first checkpoint is written, then die as a kill would."""
    save = CheckpointCallback.on_step_end

    def crash(self, state, batch=None):
        save(self, state, batch)
        raise KeyboardInterrupt

    CheckpointCallback.on_step_end = crash
    try:
        train(values)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("the run was expected to crash")
    finally:
        CheckpointCallback.on_step_end = save


def crashed_run_ends_like_an_uninterrupted_one(folder) -> None:
    whole = train(knobs(folder / "whole", **OPTIONS))

    crashed = knobs(folder / "crashed", **OPTIONS)
    run_dir = run_dir_of(crashed)
    crash_after_first_save(crashed)
    run(dict(crashed, RESUME=str(run_dir)), str(SCRIPT))

    for name in ("trainable.safetensors", "ema.safetensors"):
        assert_identical(saved(whole, name), saved(run_dir, name))


def changed_knob_is_refused_on_resume(folder) -> None:
    values = knobs(folder)
    run_dir = train(values)

    try:
        run(dict(values, LR=1.0, RESUME=str(run_dir)), str(SCRIPT))
    except SystemExit as err:
        assert "resume drift on LR" in str(err)
    else:
        raise AssertionError("a changed knob was accepted")


def failed_save_keeps_the_last_good_checkpoint(folder) -> None:
    values = knobs(folder, CKPT_EVERY=1)
    checkpoints = run_dir_of(values) / "checkpoints"
    checkpoints.mkdir(parents=True)

    # a file where step 2 wants its folder makes that write fail
    (checkpoints / "step_00000002.partial").write_text("in the way")

    train(values)
    assert latest(checkpoints).name == "step_00000001"


if __name__ == "__main__":
    run_checks(
        crashed_run_ends_like_an_uninterrupted_one,
        changed_knob_is_refused_on_resume,
        failed_save_keeps_the_last_good_checkpoint,
    )
