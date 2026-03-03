import keras
from keras import ops
from bayesflow.types import Tensor


def concatenate(tensors: list[Tensor]) -> Tensor:
    """
    Concatenate tensors of potentially different ranks by first introducing singleton
    dimensions and then tiling those before concatination.
    """
    # ensure all tensors have the same rank: [(2, 3), (2, 15, 5)] -> [(2, 1, 3), (2, 15, 5)]
    max_rank = max(t.ndim for t in tensors)
    expanded = [expand_to_target_rank(t, max_rank) for t in tensors]

    # repeat singleton dimensions for concatenation:
    # [(2, 1, 3), (2, 15, 5)] -> [(2, 15, 3), (2, 15, 5)]
    repeated = []
    for t in expanded:
        for axis in range(max_rank - 1):
            if t.shape[axis] != 1:
                continue

            dims = [x.shape[axis] for x in expanded]
            if None not in dims:
                # static: use Python max (required for jax)
                t = ops.repeat(t, max(dims), axis=axis)
            else:
                # dynamic: some dims unknown at trace time (tf graph mode)
                n = ops.shape(expanded[0])[axis]
                for x in expanded[1:]:
                    n = ops.maximum(n, ops.shape(x)[axis])
                t = ops.repeat(t, n, axis=axis)
        repeated.append(t)

    return ops.concatenate(repeated, axis=-1)


def expand_to_target_rank(tensor: Tensor, target_rank: int) -> Tensor:
    """
    Expands a tensor to a target rank by inserting singleton dimensions at axis=-2.
    """
    while tensor.ndim < target_rank:
        tensor = keras.ops.expand_dims(tensor, axis=-2)

    return tensor
