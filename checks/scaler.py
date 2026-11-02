"""Check: saving a checkpoint leaves the loss scaler as it was, and a resume restores it.

Run after changing the scaler or what a checkpoint stores.

    kogine run checks/scaler.py
"""

import torch

from latentdragon.training.scaler import DynamicScaler


def main() -> None:
    """A saving rank and a silent rank stay equal through an overflow."""
    saving, silent = DynamicScaler(poll=8), DynamicScaler(poll=8)
    state = {}
    for step in range(20):
        bad = torch.tensor(float(step == 18))
        saving.update(bad)
        silent.update(bad)
        if step == 18:
            state = saving.state_dict()
        assert saving.value == silent.value, step
        assert saving.waiting == silent.waiting, step

    # Step 19 again from the saved state: the pending overflow still backs off
    resumed = DynamicScaler(poll=8)
    resumed.load_state_dict(state)
    resumed.update(torch.tensor(0.0))
    for scaler in (saving, resumed):
        for _ in range(8):
            scaler.update(torch.tensor(0.0))
    assert resumed.value == saving.value
    assert resumed.value == DynamicScaler().value * 0.5
    print("ok scaler")


if __name__ == "__main__":
    main()
