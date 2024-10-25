import inspect
import keras
from typing import TypeVar

from collections.abc import Mapping

from bayesflow.types import Tensor

from . import logging

T = TypeVar("T")


def convert_args(f, *args, **kwargs) -> tuple[any, ...]:
    """Convert positional and keyword arguments to just positional arguments for f"""
    if not kwargs:
        return args

    signature = inspect.signature(f)

    # convert to just kwargs first
    kwargs = convert_kwargs(f, *args, **kwargs)

    parameters = []
    for name, param in signature.parameters.items():
        if param.kind in [param.VAR_POSITIONAL, param.VAR_KEYWORD]:
            continue

        parameters.append(kwargs.get(name, param.default))

    return tuple(parameters)


def convert_kwargs(f, *args, **kwargs) -> dict[str, any]:
    """Convert positional and keyword arguments to just keyword arguments for f"""
    if not args:
        return kwargs

    signature = inspect.signature(f)

    parameters = dict(zip(signature.parameters, args))

    for name, value in kwargs.items():
        if name in parameters:
            raise TypeError(f"{f.__name__}() got multiple arguments for argument '{name}'")

        parameters[name] = value

    return parameters


def filter_kwargs(kwargs: Mapping[str, T], f: callable) -> Mapping[str, T]:
    """Filter keyword arguments for f"""
    signature = inspect.signature(f)

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            # there is a **kwargs parameter, so anything is valid
            return kwargs

    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}

    return kwargs


def keras_kwargs(kwargs: Mapping[str, T]) -> dict[str, T]:
    """Keep dictionary keys that do not end with _kwargs. Used for propagating
    keyword arguments in nested layer classes.
    """
    return {key: value for key, value in kwargs.items() if not key.endswith("_kwargs")}


# TODO: rename and streamline and make protected
def check_output(outputs: T) -> None:
    # Warn if any NaNs present in output
    for k, v in outputs.items():
        nan_mask = keras.ops.isnan(v)
        if keras.ops.any(nan_mask):
            logging.warning("Found a total of {n:d} nan values for output {k}.", n=int(keras.ops.sum(nan_mask)), k=k)

    # Warn if any inf present in output
    for k, v in outputs.items():
        inf_mask = keras.ops.isinf(v)
        if keras.ops.any(inf_mask):
            logging.warning("Found a total of {n:d} inf values for output {k}.", n=int(keras.ops.sum(inf_mask)), k=k)


def split_tensors(data: Mapping[any, Tensor], axis: int = -1) -> Mapping[any, Tensor]:
    """Split tensors in the dictionary along the given axis."""
    result = {}

    for key, value in data.items():
        if keras.ops.shape(value)[axis] == 1:
            result[key] = keras.ops.squeeze(value, axis=axis)
            continue

        splits = keras.ops.split(value, keras.ops.shape(value)[axis], axis=axis)
        splits = [keras.ops.squeeze(split, axis=axis) for split in splits]

        for i, split in enumerate(splits):
            result[f"{key}_{i + 1}"] = split

    return result
