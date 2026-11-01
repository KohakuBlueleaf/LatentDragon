"""Training tasks."""

# Registry and contract
# Concrete tasks
from latentdragon.tasks.autoencoder import AutoencoderParts, AutoencoderTask
from latentdragon.tasks.base import TASK, Task

__all__ = [
    # registries
    "TASK",
    # base classes
    "Task",
    # concrete classes
    "AutoencoderParts",
    "AutoencoderTask",
]
