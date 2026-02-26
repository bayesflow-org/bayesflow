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


def expand_to_target_rank(tensor, target_rank: int):
    while tensor.ndim < target_rank:
        tensor = ops.expand_dims(tensor, axis=-2)
    return tensor


def concatenate(tensors):
    """
    Rank-align by inserting singleton dims at -2, then broadcast singleton dims
    (all axes except the last / feature axis), then concatenate on the last axis.
    """
    tensors = [ops.convert_to_tensor(t) for t in tensors]

    max_rank = max(t.ndim for t in tensors)
    expanded = [expand_to_target_rank(t, max_rank) for t in tensors]

    # runtime shapes, tuple-safe
    shapes = [_shape1d(t) for t in expanded]  # list of (rank,) int tensors
    target = ops.max(ops.stack(shapes, axis=0), axis=0)  # (rank,) = elementwise max

    # broadcast to target on all non-feature axes, keep own feature dim (last axis)
    out = []
    for t, s in zip(expanded, shapes):
        bshape = ops.concatenate([target[:-1], s[-1:]], axis=0)
        out.append(ops.broadcast_to(t, bshape))  # only works if non-feature dims are 1 or equal
    return ops.concatenate(out, axis=-1)
