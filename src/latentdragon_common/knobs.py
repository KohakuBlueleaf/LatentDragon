"""Script-owned UPPER_CASE knobs: schema from the script AST, problems collected once."""

import ast
import difflib
from pathlib import Path
from typing import Any


def script_knobs(path: str | Path) -> set[str]:
    """Every UPPER_CASE name assigned at module level of the script at `path`."""
    tree = ast.parse(Path(path).read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        match node:
            case ast.Assign(targets=targets):
                names.update(t.id for t in targets if isinstance(t, ast.Name))
            case ast.AnnAssign(target=ast.Name(id=name)):
                names.add(name)
    return {name for name in names if name.isupper() and not name.startswith("_")}


def supplied_knobs(values: dict[str, Any]) -> dict[str, Any]:
    """The public UPPER_CASE entries of a globals dict."""
    return {k: v for k, v in values.items() if k.isupper() and not k.startswith("_")}


def unknown_problems(supplied: set[str], known: set[str]) -> list[str]:
    """One message per unknown knob, with a did-you-mean hint."""
    problems = []
    for name in sorted(supplied - known):
        close = difflib.get_close_matches(name, known, n=1)
        hint = f"; did you mean {close[0]}?" if close else ""
        problems.append(f"unknown knob {name}{hint}")
    return problems


def raise_problems(problems: list[str]) -> None:
    """Exit with every problem listed; does nothing when there are none."""
    if problems:
        lines = "\n".join(f"  - {p}" for p in problems)
        raise SystemExit(f"{len(problems)} config problem(s):\n{lines}")
