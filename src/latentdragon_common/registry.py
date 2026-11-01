"""`build(choice, REGISTRY)`: the one build-time resolution point for components."""

from collections.abc import Callable
from typing import Any

from latentdragon_common.imports import import_object


class Registry:
    """Name -> factory map for one component family; dotted paths also resolve."""

    def __init__(self, family: str):
        self.family = family
        self._items: dict[str, Callable] = {}

    def register(self, name: str | None = None) -> Callable:
        """Decorator registering a class or factory under `name`."""

        def wrap(obj):
            self.add(name or obj.__name__, obj)
            return obj

        return wrap

    def add(self, name: str, obj: Callable) -> None:
        """Register `obj`; a duplicate name raises."""
        if name in self._items:
            raise KeyError(f"duplicate {self.family} '{name}'")
        self._items[name] = obj

    def get(self, name: str) -> Callable:
        """The factory for a registered name or a dotted path."""
        if name in self._items:
            return self._items[name]
        if "." in name or ":" in name:
            return import_object(name)
        known = ", ".join(sorted(self._items))
        raise KeyError(f"unknown {self.family} '{name}'; known: {known}")

    def names(self) -> list[str]:
        """Every registered name, sorted."""
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


def build(choice: Any, registry: Registry, key: str = "name", **defaults) -> Any:
    """Resolve a name, dotted path, `{key: name, ...}`, class or instance.

    `defaults` fill kwargs the choice does not set; `None` stays `None`.
    """
    match choice:
        case None:
            return None
        case str():
            return registry.get(choice)(**defaults)
        case dict():
            choice = dict(choice)
            factory = registry.get(choice.pop(key))
            return factory(**{**defaults, **choice})
        case type():
            return choice(**defaults)
        case _ if hasattr(choice, "build") and callable(choice.build):
            return choice.build(**defaults)
        case _:
            return choice
