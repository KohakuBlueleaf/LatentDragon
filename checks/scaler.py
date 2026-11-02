"""Check: the loss scaler backs off, grows and resumes on the device, with no host read per step.

Run after changing the scaler or what a checkpoint stores.

    kogine run checks/scaler.py
"""

import torch

from latentdragon.training.scaler import DynamicScaler


def check(name: str, ok: bool) -> None:
    """Print one line and stop at the first failure."""
    print(("ok  " if ok else "FAIL") + f"  {name}")
    if not ok:
        raise SystemExit(1)


def run(scaler: DynamicScaler, flags: list[float]) -> None:
    """One scale and one update per flag, as a training step does."""
    for flag in flags:
        scaler.scale(torch.ones(()))
        scaler.update(torch.tensor(flag))


def main() -> None:
    """Backoff, growth, the cap, and a resume through `state_dict`."""
    scaler = DynamicScaler(init=1024.0, interval=3)
    run(scaler, [1.0])
    check("an overflow backs off on its own step", scaler.value.item() == 512.0)
    run(scaler, [0.0, 0.0, 0.0])
    check("interval finite steps grow the scale", scaler.value.item() == 1024.0)
    run(scaler, [0.0, 1.0, 0.0, 0.0])
    check("an overflow resets the growth count", scaler.value.item() == 512.0)

    capped = DynamicScaler(init=2.0**15, interval=1)
    run(capped, [0.0, 0.0])
    check("the scale stays under the maximum", capped.value.item() == 2.0**15)

    # A save mid-count, then the same steps on the saved and the resumed scaler
    saving = DynamicScaler(init=1024.0, interval=3)
    run(saving, [0.0, 0.0])
    resumed = DynamicScaler(interval=3)
    resumed.load_state_dict(saving.state_dict())
    for scaler in (saving, resumed):
        run(scaler, [0.0, 0.0, 1.0, 0.0])
    check(
        "a resume continues the count",
        saving.value.item() == resumed.value.item() == 1024.0,
    )


if __name__ == "__main__":
    main()
