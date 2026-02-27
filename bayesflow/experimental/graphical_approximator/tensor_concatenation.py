import keras
from keras import ops
from bayesflow.types import Tensor


from keras import ops


def _shape1d(x):
    """ops.shape(x) as a 1D int tensor, even if Keras returns a Python tuple."""
    s = ops.shape(x)
    if isinstance(s, tuple):
        s = ops.stack([ops.cast(ops.convert_to_tensor(d), "int32") for d in s], axis=0)
    else:
        s = ops.cast(s, "int32")
    return s


def expand_to_target_rank(x, target_rank: int):
    while x.ndim < target_rank:
        x = ops.expand_dims(x, axis=-2)  # singleton at -2 (your convention)
    return x


def concatenate(tensors, batch_dims=1):
    if keras.backend.backend() == "tensorflow":
        return concatenate_tf(tensors, batch_dims=batch_dims)
    else:
        return concatenate_(tensors, batch_dims=batch_dims)


def concatenate_(tensors, batch_dims=1) -> Tensor:
    max_rank = max(len(t.shape) for t in tensors)

    expanded = []
    for t in tensors:
        while len(t.shape) < max_rank:
            t = keras.ops.expand_dims(t, axis=-2)
        expanded.append(t)

    # compute max shape
    max_shape = list(expanded[0].shape)
    for t in expanded[1:]:
        for i in range(batch_dims, max_rank - 1):
            max_shape[i] = max(max_shape[i], t.shape[i])

    max_shape[0] = keras.ops.shape(tensors[0])[0]
    broadcasted = []
    for t in expanded:
        # keep last dimension unique
        target = tuple(max_shape[:-1] + [t.shape[-1]])
        broadcasted.append(keras.ops.broadcast_to(t, target))

    return keras.ops.concatenate(broadcasted, axis=-1)


def concatenate_tf(tensors, batch_dims=1):
    def _shape_tensor(t):
        """
        Special case for tensorflow, because keras.ops.shape(t) returns a Python tuple.
        """
        s = keras.ops.shape(t)
        return keras.ops.convert_to_tensor(s, dtype="int32")

    max_rank = max(len(t.shape) for t in tensors)

    expanded = []
    for t in tensors:
        while len(t.shape) < max_rank:
            t = keras.ops.expand_dims(t, axis=-2)
        expanded.append(t)

    shapes = [_shape_tensor(t) for t in expanded]
    base = shapes[0]

    if max_rank - 1 > batch_dims:
        mids = keras.ops.stack(
            [s[batch_dims : max_rank - 1] for s in shapes],
            axis=0,
        )

        mids_max = keras.ops.max(mids, axis=0)
        max_shape = keras.ops.concatenate(
            [base[:batch_dims], mids_max, base[max_rank - 1 : max_rank]],
            axis=0,
        )
    else:
        max_shape = base

    broadcasted = []
    for t, s in zip(expanded, shapes):
        target = keras.ops.concatenate(
            [
                max_shape[: max_rank - 1],
                s[max_rank - 1 : max_rank],
            ],
            axis=0,
        )

        broadcasted.append(keras.ops.broadcast_to(t, target))

    return keras.ops.concatenate(broadcasted, axis=-1)
