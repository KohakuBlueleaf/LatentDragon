"""Latent augmentation, regularization and alignment slots."""

# Shared context
from latentdragon.latent.context import LatentCtx

# Teachers, terms and views with their registries; built-in terms register on import
from latentdragon.latent.state_terms import StateRms
from latentdragon.latent.teachers import TEACHER, Teacher
from latentdragon.latent.terms import LATENT_TERM, WEIGHTING, LatentTerm
from latentdragon.latent.views import LATENT_VIEW, LatentView

__all__ = [
    # registries
    "LATENT_TERM",
    "LATENT_VIEW",
    "TEACHER",
    "WEIGHTING",
    # base classes
    "LatentCtx",
    "LatentTerm",
    "LatentView",
    "Teacher",
    # built-in terms
    "StateRms",
]
