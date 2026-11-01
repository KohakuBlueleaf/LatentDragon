"""Dotted-path resolution for user-supplied components."""

import functools
import importlib


def import_object(path: str):
    """`pkg.module:Attr.sub` (explicit module) or `pkg.module.Attr.sub` (the longest
    importable prefix is the module) -> the attribute chain."""
    # Explicit module part
    if ":" in path:
        module, _, chain = path.partition(":")
        return functools.reduce(
            getattr, chain.split("."), importlib.import_module(module)
        )

    # Try the longest module prefix first; a missing prefix itself means try shorter
    parts = path.split(".")
    for cut in range(len(parts) - 1, 0, -1):
        prefix = ".".join(parts[:cut])
        try:
            module = importlib.import_module(prefix)
        except ModuleNotFoundError as err:
            if err.name is None or not f"{prefix}.".startswith(f"{err.name}."):
                raise
            continue
        return functools.reduce(getattr, parts[cut:], module)
    raise ImportError(f"no importable module prefix in {path!r}")
