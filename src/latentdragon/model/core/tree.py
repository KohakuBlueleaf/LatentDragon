"""Slot trees: the batch-first tensors that cross every arrow, in any nesting.

A sample, a latent and an output are each a dict of named slots; a slot holds a tensor or any
pytree of tensors. The one structural rule is that every tensor is batch-first, which is what
lets the trainer move, split and stack batches without knowing their shape.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch import Tensor
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

__all__ = [
    "batch_size",
    "cat",
    "chunk",
    "dot",
    "grad",
    "leaves",
    "like",
    "map_tensors",
    "named_tensors",
    "stack",
    "to_device",
]


def leaves(tree: Any) -> list[Tensor]:
    """Every tensor of `tree`, in a fixed order."""
    return [leaf for leaf in tree_flatten(tree)[0] if isinstance(leaf, Tensor)]


def like(tree: Any, values: Sequence[Tensor]) -> Any:
    """`tree` with its tensors replaced, in `leaves` order, by `values`."""
    flat, spec = tree_flatten(tree)
    values = iter(values)
    swapped = [next(values) if isinstance(leaf, Tensor) else leaf for leaf in flat]
    return tree_unflatten(swapped, spec)


def map_tensors(fn: Callable[[Tensor], Any], tree: Any) -> Any:
    """`fn` applied to every tensor; other leaves pass through."""
    return tree_map(lambda leaf: fn(leaf) if isinstance(leaf, Tensor) else leaf, tree)


def named_tensors(tree: Any, prefix: str = "") -> dict[str, Tensor]:
    """Every tensor keyed by its dotted path, e.g. `{"video": t}` or `{"z.0": t}`."""
    match tree:
        case Tensor():
            return {prefix: tree}
        case dict():
            items = tree.items()
        case list() | tuple():
            items = enumerate(tree)
        case _:
            return {}
    out = {}
    for key, value in items:
        path = f"{prefix}.{key}" if prefix else str(key)
        out.update(named_tensors(value, path))
    return out


def batch_size(tree: Any) -> int:
    """The batch dimension of the first tensor."""
    return leaves(tree)[0].shape[0]


def chunk(tree: Any, count: int) -> list[Any]:
    """`count` micro-batches along dim 0 of every tensor."""
    parts = [leaf.chunk(count, dim=0) for leaf in leaves(tree)]
    return [like(tree, [part[index] for part in parts]) for index in range(count)]


def cat(trees: Sequence[Any]) -> Any:
    """Batches joined along dim 0."""
    columns = zip(*(leaves(tree) for tree in trees), strict=True)
    return like(trees[0], [torch.cat(column, dim=0) for column in columns])


def stack(trees: Sequence[Any]) -> Any:
    """Samples stacked into a batch."""
    columns = zip(*(leaves(tree) for tree in trees), strict=True)
    return like(trees[0], [torch.stack(column, dim=0) for column in columns])


def to_device(tree: Any, device, image_format=torch.preserve_format) -> Any:
    """Every tensor on `device`; 4D tensors take `image_format`."""

    def move(leaf: Tensor) -> Tensor:
        layout = image_format if leaf.ndim == 4 else torch.preserve_format
        return leaf.to(device, non_blocking=True, memory_format=layout)

    return map_tensors(move, tree)


def dot(left: Any, right: Any) -> Tensor:
    """Sum over every tensor pair of `(left * right).sum()`."""
    pairs = zip(leaves(left), leaves(right), strict=True)
    total = None
    for a, b in pairs:
        term = (a * b).sum()
        total = term if total is None else total + term
    return total


def grad(
    outputs: Any,
    inputs: Any,
    grad_outputs: Any = None,
    retain_graph: bool = False,
    create_graph: bool = False,
) -> Any:
    """`torch.autograd.grad` over trees; the result has the shape of `inputs`.

    An input the outputs never read gets zeros: a critic reading one slot of a multi-slot
    output still yields a gradient for every slot.
    """
    outs = [outputs] if isinstance(outputs, Tensor) else leaves(outputs)
    seeds = None
    if grad_outputs is not None:
        seeds = (
            [grad_outputs] if isinstance(grad_outputs, Tensor) else leaves(grad_outputs)
        )
    wrt = [inputs] if isinstance(inputs, Tensor) else leaves(inputs)
    found = torch.autograd.grad(
        outs,
        wrt,
        grad_outputs=seeds,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=True,
        materialize_grads=True,
    )
    return found[0] if isinstance(inputs, Tensor) else like(inputs, found)
