"""A Lightning-style module summary: one row per module down to a depth, then totals."""

from torch import nn

COLUMNS = ("", "Name", "Type", "Params", "Trainable", "Mode")


def readable(count: int) -> str:
    """`count` as Lightning prints it: plain below a thousand, else `1.2 M`."""
    if count < 1000:
        return str(count)
    for label in ("K", "M", "B", "T"):
        count /= 1000
        if count < 1000 or label == "T":
            return f"{count:,.1f} {label}"
    return str(count)


def _rows(modules: dict[str, nn.Module], depth: int) -> list[tuple]:
    """`(name, type, params, trainable, mode)` for every module down to `depth` dots."""
    rows = []
    for top, module in modules.items():
        for name, part in module.named_modules(prefix=top):
            if name.count(".") >= depth:
                continue
            params = list(part.parameters())
            rows.append(
                (
                    name,
                    type(part).__name__,
                    sum(p.numel() for p in params),
                    sum(p.numel() for p in params if p.requires_grad),
                    "train" if part.training else "eval",
                )
            )
    return rows


def _unique(modules: dict[str, nn.Module]) -> list[nn.Parameter]:
    """Every parameter once, so a weight shared by two modules counts once in the totals."""
    seen = {}
    for module in modules.values():
        for param in module.parameters():
            seen[id(param)] = param
    return list(seen.values())


def summarize(modules: dict[str, nn.Module], depth: int) -> tuple[str, dict]:
    """The table text and its totals, for the console, the run folder and wandb."""
    rows = _rows(modules, depth)
    params = _unique(modules)
    every = [m for module in modules.values() for m in module.modules()]
    totals = {
        "params_trainable": sum(p.numel() for p in params if p.requires_grad),
        "params_frozen": sum(p.numel() for p in params if not p.requires_grad),
        "params_total": sum(p.numel() for p in params),
        "params_mb": sum(p.numel() * p.element_size() for p in params) / 1e6,
        "modules_train": sum(m.training for m in every),
        "modules_eval": sum(not m.training for m in every),
    }

    # Column widths from the header and every cell
    cells = [
        (str(i), name, kind, readable(count), readable(trainable), mode)
        for i, (name, kind, count, trainable, mode) in enumerate(rows)
    ]
    widths = [max(len(row[col]) for row in [COLUMNS, *cells]) for col in range(6)]
    lines = [" | ".join(c.ljust(w) for c, w in zip(row, widths)) for row in cells]
    header = " | ".join(c.ljust(w) for c, w in zip(COLUMNS, widths))
    rule = "-" * len(header)
    footer = [
        f"{readable(totals['params_trainable']):<10}Trainable params",
        f"{readable(totals['params_frozen']):<10}Non-trainable params",
        f"{readable(totals['params_total']):<10}Total params",
        f"{totals['params_mb']:<10,.3f}Total estimated model params size (MB)",
        f"{totals['modules_train']:<10}Modules in train mode",
        f"{totals['modules_eval']:<10}Modules in eval mode",
    ]
    table = [header, rule, *lines, rule, *footer]
    return "\n".join(line.rstrip() for line in table), totals
