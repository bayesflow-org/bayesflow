from functools import reduce

import sympy as sp

Dim = int | sp.Expr
Shape = tuple[Dim, ...]


def concatenate_shapes(shapes: list[Shape]) -> Shape:
    """
    Combine multiple shapes into a single stacked shape.
    All input shapes are first expanded to the same rank, then merged
    by iteratively stacking them along the last axis.
    """
    max_rank = max(len(s) for s in shapes)
    expanded = [expand_shape_rank(s, max_rank) for s in shapes]

    return reduce(stack_shapes, expanded)


def resolve_shapes(x: dict[str, Shape], meta_dict: dict | None) -> dict[str, Shape]:
    """
    Replaces string placeholders in a dictionary of shape tuples.
    >>> resolve_meta({"beta": ("B", "N_groups", 3)}, {"B": 10, "N_groups": 15})
    {'beta': (10, 15, 3)}
    """

    return {k: replace_placeholders(v, meta_dict) for k, v in x.items()}


def replace_placeholders(shape: Shape, meta_dict: dict | None = None) -> Shape:
    """
    Replaces string placeholders in shape tuples.
    >>> replace_placeholders(("B", "N", 3), {"B": 4, "N": 3})
    (4, 3, 3)
    """

    meta_dict = meta_dict or {}

    def resolve(x: sp.Expr | int) -> int | sp.Expr:
        if isinstance(x, sp.Expr):
            x = sp.simplify(x.subs(meta_dict))
            if isinstance(x, sp.Integer):
                x = int(x)

            return x
        else:
            return x

    return tuple(resolve(x) for x in shape)


def stack_shapes(a: Shape, b: Shape, axis=-1) -> Shape:
    """
    Stacks two shape tuples along an axis. The size along the staking axis
    is the sum of both shapes, while all other dimensions take the maximum of
    the two.
    >>> stack_shapes((2, 20), (2, 3, 3), axis=-1)
    (2, 3, 23)
    """

    def is_one(x: Dim) -> bool:
        return x == 1 if isinstance(x, int) else sp.simplify(x - 1) == 0

    def max_dim(x: Dim, y: Dim) -> Dim:
        if x == y:
            return x
        elif not is_one(x) and is_one(y):
            return x
        elif not is_one(y) and is_one(x):
            return y
        else:
            raise ValueError(f"Cannot stack shapes with differing placeholders {a[i]} and {b[i]}.")

    rank = max(len(a), len(b))
    axis = axis % rank

    a = expand_shape_rank(a, rank)
    b = expand_shape_rank(b, rank)

    output = []
    for i in range(rank):
        output.append(a[i] + b[i] if i == axis else max_dim(a[i], b[i]))

    simplified = [sp.simplify(x) for x in output]
    output = [int(x) if isinstance(x, sp.Integer) else x for x in simplified]

    return tuple(output)


def expand_shape_rank(shape: tuple[int | sp.Expr, ...], target_rank: int) -> tuple[int | sp.Expr, ...]:
    """
    Expand a tensor shape to a desired rank by inserting singleton (1)
    dimensions immediately before the last dimension.
    >>> expand_shape_rank((10, 2, 3), 5)
    (10, 2, 1, 1, 3)
    """

    rank = len(shape)
    if target_rank < rank:
        raise ValueError(f"Shape rank {len(shape)} must be less than target_rank {target_rank}.")

    inserts = target_rank - rank

    return shape[:-1] + (1,) * inserts + shape[-1:]
