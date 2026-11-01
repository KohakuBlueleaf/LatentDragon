"""Teachers: frozen feature models, built once and never checkpointed."""

import torch

from latentdragon_common.registry import Registry

TEACHER = Registry("teacher")


class Teacher(torch.nn.Module):
    """`features(x)` under no_grad; the build site calls `freeze`."""

    def freeze(self) -> "Teacher":
        """Disable gradients and switch to eval mode."""
        self.requires_grad_(False)
        return self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map the signal to teacher features; subclasses implement it."""
        raise NotImplementedError

    @torch.no_grad()
    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The forward features without autograd tracking."""
        return self(x)
