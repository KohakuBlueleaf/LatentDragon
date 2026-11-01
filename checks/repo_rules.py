"""Repo rules: file length, ASCII, in-function imports, import direction."""

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_LINES = 1000
SOURCE_DIRS = ("src", "scripts", "checks", "configs", "experiments")
ALLOWED_IMPORTS = {
    "latentdragon_common": ("latentdragon_common",),
    "latentdragon_kernels": ("latentdragon_common", "latentdragon_kernels"),
    "latentdragon_data": ("latentdragon_common", "latentdragon_data"),
}
PURE_PACKAGES = ("latentdragon.model", "latentdragon.vq", "latentdragon.bottleneck")


def python_files(dirs: tuple[str, ...] = SOURCE_DIRS) -> list[Path]:
    """Every Python file under `dirs`."""
    return [p for d in dirs for p in sorted((ROOT / d).rglob("*.py"))]


def module_name(path: Path) -> str:
    """The dotted module path of a file under `src`."""
    relative = path.relative_to(ROOT / "src").with_suffix("")
    return ".".join(relative.parts).removesuffix(".__init__")


def nested_imports(text: str) -> list[int]:
    """Lines of imports inside a function body, outside a `try` guard."""
    found: list[int] = []

    def walk(node: ast.AST, in_function: bool, in_try: bool) -> None:
        for child in ast.iter_child_nodes(node):
            function = in_function or isinstance(
                child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
            )
            guarded = in_try or isinstance(child, ast.Try)
            is_import = isinstance(child, (ast.Import, ast.ImportFrom))
            if is_import and function and not guarded:
                found.append(child.lineno)
            walk(child, function, guarded)

    walk(ast.parse(text), False, False)
    return found


def imported_modules(text: str) -> set[str]:
    """Absolute module names imported anywhere in `text`."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(text)):
        match node:
            case ast.Import(names=names):
                modules.update(alias.name for alias in names)
            case ast.ImportFrom(module=str(module), level=0):
                modules.add(module)
    return modules


def direction_problems(package: str, imports: set[str]) -> list[str]:
    """Imports that break the package direction of the dotted module `package`."""
    top = package.split(".")[0]
    problems = []
    for name in sorted(imports):
        root = name.split(".")[0]

        # sibling packages only in the allowed direction
        if root.startswith("latentdragon") and root not in ALLOWED_IMPORTS.get(
            top, (root,)
        ):
            problems.append(f"{package} imports {name}")

        # model, vq and bottleneck stay free of training
        if package.startswith(PURE_PACKAGES) and name.startswith(
            "latentdragon.training"
        ):
            problems.append(f"{package} imports {name}")

        # common stays free of torch
        if top == "latentdragon_common" and root == "torch":
            problems.append(f"{package} imports {name}")
    return problems


def file_problems(path: Path) -> list[str]:
    """Length and ASCII problems of one file, and its in-function imports."""
    text = path.read_text(encoding="utf-8")
    problems = []

    lines = text.count("\n")
    if lines > MAX_LINES:
        problems.append(f"{path}: {lines} lines")

    if not text.isascii():
        problems.append(f"{path}: non-ASCII text")

    if path.is_relative_to(ROOT / "src") or path.is_relative_to(ROOT / "scripts"):
        problems += [f"{path}:{n}: in-function import" for n in nested_imports(text)]
    return problems


def main() -> None:
    problems = [issue for path in python_files() for issue in file_problems(path)]
    for path in python_files(("src",)):
        imports = imported_modules(path.read_text(encoding="utf-8"))
        problems += direction_problems(module_name(path), imports)

    if problems:
        raise SystemExit("\n".join(problems))
    print("ok  repo rules")


if __name__ == "__main__":
    main()
