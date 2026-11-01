"""Critics: importing this package registers every built-in."""

# Registry and contract
from latentdragon.gan.critics.base import CRITIC, Critic

# Concrete critics
from latentdragon.gan.critics.modern import ModernPatchGAN
from latentdragon.gan.critics.patchgan import PatchGAN

__all__ = [
    # registries
    "CRITIC",
    # base classes
    "Critic",
    # concrete classes
    "ModernPatchGAN",
    "PatchGAN",
]
