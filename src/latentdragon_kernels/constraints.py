"""`requires`: the declarative `supports(arch)` builder shared by backends."""

from collections.abc import Callable

from latentdragon_kernels.hardware.arch import kind_of, sm_of


def requires(
    kinds: tuple[str, ...] = ("cuda",),
    min_sm: int = 0,
    max_sm: int = 10_000,
) -> Callable[[str], str | None]:
    """A `supports` function: the reason an arch tag is rejected, else None."""

    def supports(arch: str) -> str | None:
        if kind_of(arch) not in kinds:
            return f"device kind {kind_of(arch)} not in {kinds}"
        if kind_of(arch) == "cuda" and not min_sm <= sm_of(arch) <= max_sm:
            return f"{arch} outside sm{min_sm}..sm{max_sm}"
        return None

    return supports
