from collections.abc import Callable, Mapping, Sequence
import keras
import numpy as np

from bayesflow.types import Shape


def batched_call(
    f: callable,
    batch_shape: Shape,
    args: Sequence[any] = (),
    kwargs: Mapping[str, any] = None,
    map_predicate: Callable[[any], bool] = None,
    flatten: bool = False,
) -> list:
    """Map f over the given batch shape with a for loop, preserving randomness unlike the keras built-in map apis.

    :param f: The function to call.

    :param batch_shape: The shape of the batch.

    :param args: Any number and type of positional arguments to f.
        Arguments indicated by `map_predicate` will be indexed over the first len(batch_shape) axes.

    :param kwargs: Any number and type of keyword arguments to f.
        Arguments indicated by `map_predicate` will be indexed over the first len(batch_shape) axes.

    :param map_predicate: A function that returns True if an argument should be indexed over the batch shape.
        By default, all array-like arguments are mapped.

    :param flatten: Whether to flatten the output.

    :return: A list of outputs of f for each element in the batch.
    """
    if kwargs is None:
        kwargs = {}

    if map_predicate is None:

        def map_predicate(arg):
            if isinstance(arg, np.ndarray):
                return arg.ndim >= len(batch_shape)
            if keras.ops.is_tensor(arg):
                return keras.ops.ndim(arg) >= len(batch_shape)

            return False

    outputs = np.empty(batch_shape, dtype="object")

    for index in np.ndindex(batch_shape):
        map_args = [arg[index] if map_predicate(arg) else arg for arg in args]
        map_kwargs = {key: value[index] if map_predicate(value) else value for key, value in kwargs.items()}

        outputs[index] = f(*map_args, **map_kwargs)

    if flatten:
        outputs = outputs.flatten()

    return outputs.tolist()
