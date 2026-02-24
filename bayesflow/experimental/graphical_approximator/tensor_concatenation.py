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

    # maximum for each dimension: [(2, 1, 3), (2, 15, 5)] -> (2, 15, 5)
    max_shape = ()
    for dims in zip(*(x.shape for x in expanded)):
        max_dim = max((x for x in dims if x is not None), default=None)
        max_shape = max_shape + (max_dim,)

    # repeat singleton dimensions for concatenation:
    # [(2, 1, 3), (2, 15, 5)] -> [(2, 15, 3), (2, 15, 5)]
    repeated = []
    for t in expanded:
        for axis, (dim, target) in enumerate(zip(t.shape[:-1], max_shape[:-1])):
            if target is None or dim == target:  # skipping None in graph mode
                continue
            t = ops.repeat(t, repeats=target, axis=axis)
        repeated.append(t)

    return ops.concatenate(repeated, axis=-1)


def expand_to_target_rank(tensor: Tensor, target_rank: int) -> Tensor:
    """
    Expands a tensor to a target rank by inserting singleton dimensions at axis=-2.
    """
    while tensor.ndim < target_rank:
        tensor = keras.ops.expand_dims(tensor, axis=-2)

    return tensor
