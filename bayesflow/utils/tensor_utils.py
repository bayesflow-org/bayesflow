from collections.abc import Sequence
from typing import TypeVar

import keras
import numpy as np

from bayesflow.types import Tensor, Shape
from . import logging

T = TypeVar("T")


def concatenate_valid(tensors: Sequence[Tensor | None], axis: int = 0) -> Tensor | None:
    """Concatenate multiple tensors along axis, ignoring None values."""
    tensors = [t for t in tensors if t is not None]

    if not tensors:
        return None

    return keras.ops.concatenate(tensors, axis=axis)


def concatenate_valid_shapes(tensor_shapes: Sequence[Shape | None], axis: int = 0) -> Shape | None:
    tensor_shapes = [s for s in tensor_shapes if s is not None]
    if not tensor_shapes:
        return None

    output_shape = list(tensor_shapes[0])
    for s in tensor_shapes[1:]:
        output_shape[axis] += s[axis]
    return output_shape


def expand(x: Tensor, n: int, side: str):
    if n < 0:
        raise ValueError(f"Cannot expand {n} times.")

    match side:
        case "left":
            idx = [None] * n + [...]
        case "right":
            idx = [...] + [None] * n
        case str() as name:
            raise ValueError(f"Invalid side {name!r}. Must be 'left' or 'right'.")
        case other:
            raise TypeError(f"Invalid side type {type(other)!r}. Must be str.")

    return x[tuple(idx)]


def expand_as(x: Tensor, y: Tensor, side: str):
    return expand_to(x, keras.ops.ndim(y), side)


def expand_to(x: Tensor, dim: int, side: str):
    return expand(x, dim - keras.ops.ndim(x), side)


def expand_left(x: Tensor, n: int) -> Tensor:
    """Expand x to the left n times"""
    if n < 0:
        raise ValueError(f"Cannot expand {n} times.")

    idx = [None] * n + [...]
    return x[tuple(idx)]


def expand_left_as(x: Tensor, y: Tensor) -> Tensor:
    """Expand x to the left, matching the dimension of y"""
    return expand_left_to(x, keras.ops.ndim(y))


def expand_left_to(x: Tensor, dim: int) -> Tensor:
    """Expand x to the left, matching dim"""
    return expand_left(x, dim - keras.ops.ndim(x))


def expand_right(x: Tensor, n: int) -> Tensor:
    """Expand x to the right n times"""
    if n < 0:
        raise ValueError(f"Cannot expand {n} times.")

    idx = [...] + [None] * n
    return x[tuple(idx)]


def expand_right_as(x: Tensor, y: Tensor) -> Tensor:
    """Expand x to the right, matching the dimension of y"""
    return expand_right_to(x, keras.ops.ndim(y))


def expand_right_to(x: Tensor, dim: int) -> Tensor:
    """Expand x to the right, matching dim"""
    return expand_right(x, dim - keras.ops.ndim(x))


def expand_tile(x: Tensor, n: int, axis: int) -> Tensor:
    """Expand and tile x along the given axis n times"""
    if keras.ops.is_tensor(x):
        x = keras.ops.expand_dims(x, axis)
    else:
        x = np.expand_dims(x, axis)

    return tile_axis(x, n, axis=axis)


def is_symbolic_tensor(x: Tensor) -> bool:
    if keras.utils.is_keras_tensor(x):
        return True

    match keras.backend.backend():
        case "jax":
            import jax

            return not jax.core.is_concrete(x)
        case "tensorflow":
            import tensorflow as tf

            return tf.is_symbolic_tensor(x)
        case "torch":
            return False
        case _:
            raise NotImplementedError(f"Symbolic tensor check not implemented for backend {keras.backend.backend()!r}")


def pad(x: Tensor, value: float | Tensor, n: int, axis: int, side: str = "both") -> Tensor:
    """
    Pad x with n values along axis on the given side.
    The pad value must broadcast against the shape of x, except for the pad axis, where it must broadcast against n.
    """
    if not keras.ops.is_tensor(value):
        value = keras.ops.full((), value, dtype=keras.ops.dtype(x))

    shape = list(keras.ops.shape(x))
    shape[axis] = n

    p = keras.ops.broadcast_to(value, shape)
    match side:
        case "left":
            return keras.ops.concatenate([p, x], axis=axis)
        case "right":
            return keras.ops.concatenate([x, p], axis=axis)
        case "both":
            return keras.ops.concatenate([p, x, p], axis=axis)
        case str() as name:
            raise ValueError(f"Invalid side {name!r}. Must be 'left', 'right', or 'both'.")
        case _:
            raise TypeError(f"Invalid side type {type(side)!r}. Must be str.")


def weighted_mean(elements: Tensor, weights: Tensor = None) -> Tensor:
    """
    Compute the (optionally) weighted mean of the input tensor.

    Parameters
    ----------
    elements : Tensor
        A tensor containing the elements to average.
    weights : Tensor, optional
        A tensor of the same shape as `elements` representing weights.
        If None, the mean is computed without weights.

    Returns
    -------
    Tensor
        A scalar tensor representing the (weighted) mean.
    """
    return keras.ops.mean(elements * weights if weights is not None else elements)


def searchsorted(sorted_sequence: Tensor, values: Tensor, side: str = "left") -> Tensor:
    """
    Find indices where elements should be inserted to maintain order.
    """

    match keras.backend.backend():
        case "jax":
            import jax
            import jax.numpy as jnp

            logging.warn_once(f"searchsorted is not yet optimized for backend {keras.backend.backend()!r}")

            # do not vmap over the side argument (we have to pass it as a positional argument)
            in_axes = [0, 0, None]

            # vmap over the batch dimension
            vss = jax.vmap(jnp.searchsorted, in_axes=in_axes)

            # flatten all batch dimensions
            ss = sorted_sequence.reshape((-1,) + sorted_sequence.shape[-1:])
            v = values.reshape((-1,) + values.shape[-1:])

            # noinspection PyTypeChecker
            indices = vss(ss, v, side)

            # restore the batch dimensions
            indices = indices.reshape(values.shape)

            # noinspection PyTypeChecker
            return indices
        case "tensorflow":
            import tensorflow as tf

            # always use int64 to avoid complicated graph code
            indices = tf.searchsorted(sorted_sequence, values, side=side, out_type="int64")

            return indices
        case "torch":
            import torch

            out_int32 = len(sorted_sequence) <= np.iinfo(np.int32).max

            indices = torch.searchsorted(
                sorted_sequence.contiguous(), values.contiguous(), side=side, out_int32=out_int32
            )

            return indices
        case _:
            raise NotImplementedError(f"Searchsorted not implemented for backend {keras.backend.backend()!r}")


def size_of(x) -> int:
    """
    :param x: A nested structure of tensors.
    :return: The total memory footprint of x, ignoring view semantics, in bytes.
    """
    if isinstance(x, (int, float)):
        x = np.asarray(x)
    if keras.ops.is_tensor(x) or isinstance(x, np.ndarray):
        return int(keras.ops.size(x)) * np.dtype(keras.ops.dtype(x)).itemsize

    # flatten nested structure
    x = keras.tree.flatten(x)

    # get unique tensors by id
    x = {id(tensor): tensor for tensor in x}

    # sum up individual sizes
    return sum(size_of(tensor) for tensor in x.values())


def stack_valid(tensors: Sequence[Tensor | None], axis: int = 0) -> Tensor | None:
    """Stack multiple tensors along axis, ignoring None values."""
    tensors = [t for t in tensors if t is not None]

    if not tensors:
        return None

    return keras.ops.stack(tensors, axis=axis)


def tile_axis(x: Tensor, n: int, axis: int) -> Tensor:
    """Tile x along the given axis n times"""
    repeats = [1] * keras.ops.ndim(x)
    repeats[axis] = n

    if keras.ops.is_tensor(x):
        return keras.ops.tile(x, repeats)

    return np.tile(x, repeats)


# we want to annotate this as Sequence[PyTree[Tensor]], but static type checkers do not support PyTree's type expansion
def tree_concatenate(structures: Sequence[T], axis: int = 0, numpy: bool = None) -> T:
    """Concatenate all tensors in the given sequence of nested structures.
    All objects in the given sequence must have the same structure.
    The output will adhere to this structure.

    :param structures: A sequence of nested structures of tensors.
        All structures in the sequence must have the same layout.
        Tensors in the same layout location must have compatible shapes for concatenation.
    :param axis: The axis along which to concatenate tensors.
    :param numpy: Whether to use numpy or keras for concatenation.
        Will convert all items in the structures to numpy arrays if True, tensors otherwise.
        Defaults to True if all tensors are numpy arrays, False otherwise.
    :return: A structure of concatenated tensors with the same layout as each input structure.
    """
    if numpy is None:
        numpy = not any(keras.tree.flatten(keras.tree.map_structure(keras.ops.is_tensor, structures)))

    if numpy:
        structures = keras.tree.map_structure(keras.ops.convert_to_numpy, structures)

        def concat(*items):
            return np.concatenate(items, axis=axis)
    else:
        structures = keras.tree.map_structure(keras.ops.convert_to_tensor, structures)

        def concat(*items):
            return keras.ops.concatenate(items, axis=axis)

    return keras.tree.map_structure(concat, *structures)


def tree_stack(structures: Sequence[T], axis: int = 0, numpy: bool = None) -> T:
    """Like :func:`tree_concatenate`, except tensors are stacked instead of concatenated."""
    if numpy is None:
        numpy = not any(keras.tree.flatten(keras.tree.map_structure(keras.ops.is_tensor, structures)))

    if numpy:
        structures = keras.tree.map_structure(keras.ops.convert_to_numpy, structures)

        def stack(*items):
            return np.stack(items, axis=axis)
    else:
        structures = keras.tree.map_structure(keras.ops.convert_to_tensor, structures)

        def stack(*items):
            return keras.ops.stack(items, axis=axis)

    return keras.tree.map_structure(stack, *structures)


def fill_triangular_matrix(x: Tensor, upper: bool = False, positive_diag: bool = False):
    """
    Reshapes a batch of matrix elements into a triangular matrix (either upper or lower).

    Note: If final axis has length 1, this simply reshapes to (batch_size, 1, 1) and optionally applies softplus.

    Parameters
    ----------
    x : Tensor of shape (batch_size, m)
        Batch of flattened nonzero matrix elements for triangular matrix.
    upper : bool
        Return upper triangular matrix if True, else lower triangular matrix. Default is False.

    Returns
    -------
    Tensor of shape (batch_size, n, n)
        Batch of triangular matrices with m = n * (n + 1) / 2 unique nonzero elements.

    Raises
    ------
    ValueError
        If provided nonzero elements do not correspond to possible triangular matrix shape
        (n,n) with n = sqrt( 1/4 + 2 * m) - 1/2 due to m = n * (n + 1) / 2.
    """
    batch_shape = x.shape[:-1]
    m = x.shape[-1]

    if m > 1:  # Matrix is larger than than 1x1
        # Calculate matrix shape
        n = (0.25 + 2 * m) ** 0.5 - 0.5
        if not np.isclose(np.floor(n), n):
            raise ValueError(f"Input right-most shape ({m}) does not correspond to a triangular matrix.")
        else:
            n = int(n)

        # Trick: Create triangular matrix by concatenating with a flipped version of itself, then reshape.
        if not upper:
            x_list = [x[..., n:], keras.ops.flip(x, axis=-1)]

            y = keras.ops.concatenate(x_list, axis=len(batch_shape))
            y = keras.ops.reshape(y, (-1, n, n))
            y = keras.ops.tril(y)

        else:
            x_list = [x, keras.ops.flip(x[..., n:], axis=-1)]

            y = keras.ops.concatenate(x_list, axis=len(batch_shape))
            y = keras.ops.reshape(y, (-1, n, n))
            y = keras.ops.triu(
                y,
            )

    else:  # Matrix is 1x1
        y = keras.ops.reshape(x, (-1, 1, 1))

    return y


def positive_diag(x: Tensor, method="default") -> Tensor:
    """
    Ensures that matrix elements on diagonal are positive.

    Parameters
    ----------
    x : Tensor of shape (batch_size, n, n)
        Batch of matrices.
    method : str, optional
        Method by which to ensure positivity of diagonal entries. Choose from
        - "shifted_softplus": softplus(x + 0.5413)
        - "exp": exp(x)
        Both methods map a matrix filled with zeros to the unit matrix.
        Default is "shifted_softplus".

    Returns
    -------
    Tensor of shape (batch_size, n, n)
    """
    # ensure positivity
    match method:
        case "default" | "shifted_softplus":
            x_positive = keras.activations.softplus(x + 0.5413) + 1e-5
        case "exp":
            x_positive = keras.ops.exp(x) + 1e-5

    # zero all offdiagonals
    x_diag_positive = keras.ops.tril(keras.ops.triu(x_positive))

    # zero diagonal entries
    x_offdiag = keras.ops.triu(x, k=1) + keras.ops.tril(x, k=-1)

    # sum to get full matrices with softplus applied only to diagonal entries
    x = x_diag_positive + x_offdiag

    return x
