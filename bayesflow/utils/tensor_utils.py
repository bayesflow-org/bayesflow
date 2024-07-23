import keras

from bayesflow.types import Shape, Tensor


def broadcast_right(x: Tensor, shape: Shape) -> Tensor:
    """Broadcast x to the given shape, expanding to the right as necessary"""
    return keras.ops.broadcast_to(expand_right_to(x, len(shape)), shape)


def broadcast_right_as(x: Tensor, y: Tensor) -> Tensor:
    """Broadcast x to the shape of y, expanding to the right as necessary"""
    return broadcast_right(x, keras.ops.shape(y))


def expand_right(x: Tensor, n: int) -> Tensor:
    """Expand x to the right n times"""
    if n < 0:
        raise ValueError(f"Cannot expand {n} times.")

    idx = [...] + [None] * n
    return x[tuple(idx)]


def expand_right_to(x: Tensor, dim: int) -> Tensor:
    """Expand x to the right, matching dim"""
    return expand_right(x, dim - keras.ops.ndim(x))


def expand_right_as(x: Tensor, y: Tensor) -> Tensor:
    """Expand x to the right, matching the dimension of y"""
    return expand_right_to(x, keras.ops.ndim(y))


def expand_tile(x: Tensor, axis: int, n: int) -> Tensor:
    """Expand and tile x along the given axis n times"""
    x = keras.ops.expand_dims(x, axis=axis)
    return tile_axis(x, axis, n)


def tile_axis(x: Tensor, axis: int, n: int) -> Tensor:
    """Tile x along the given axis n times"""
    repeats = [1] * keras.ops.ndim(x)
    repeats[axis] = n
    return keras.ops.tile(x, repeats=repeats)
