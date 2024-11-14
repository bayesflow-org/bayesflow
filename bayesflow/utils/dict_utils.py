import inspect
import keras
from typing import TypeVar

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from bayesflow.types import Tensor

from . import logging

T = TypeVar("T")


def convert_args(f: Callable, *args: any, **kwargs: any) -> tuple[any, ...]:
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


def convert_kwargs(f: Callable, *args: any, **kwargs: any) -> dict[str, any]:
    """Convert positional and keyword arguments qto just keyword arguments for f"""
    if not args:
        return kwargs

    signature = inspect.signature(f)

    parameters = dict(zip(signature.parameters, args))

    for name, value in kwargs.items():
        if name in parameters:
            raise TypeError(f"{f.__name__}() got multiple arguments for argument '{name}'")

        parameters[name] = value

    return parameters


def filter_kwargs(kwargs: Mapping[str, T], f: Callable) -> Mapping[str, T]:
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


def dicts_to_arrays(
    post_variables: dict[str, np.ndarray] | np.ndarray,
    prior_variables: dict[str, np.ndarray] | np.ndarray,
    names: Sequence[str] = None,
    context: str = None,
):
    """Utility to optionally convert dicts as returned from approximators and adapters into arrays."""

    if type(post_variables) is not type(prior_variables):
        raise ValueError("You should either use dicts or tensors, but not separate types for your inputs.")

    if isinstance(post_variables, dict):
        if post_variables.keys() != prior_variables.keys():
            raise ValueError("Keys in your posterior / prior arrays should match.")

        # Use user-provided names instead of inferred ones
        names = list(post_variables.keys()) if names is None else names

        post_variables = np.concatenate([v for k, v in post_variables.items() if k in names], axis=-1)
        prior_variables = np.concatenate([v for k, v in prior_variables.items() if k in names], axis=-1)

    elif isinstance(post_variables, np.ndarray):
        if names is not None:
            if post_variables.shape[-1] != len(names) or prior_variables.shape[-1] != len(names):
                raise ValueError("The length of the names list should match the number of target variables.")
        else:
            if context is not None:
                names = [f"${context}_{{{i}}}$" for i in range(post_variables.shape[-1])]
            else:
                names = [f"$\\theta_{{{i}}}$" for i in range(post_variables.shape[-1])]

    else:
        raise TypeError("Only dicts and tensors are supported as arguments.")

    return dict(post_variables=post_variables, prior_variables=prior_variables, names=names, num_variables=len(names))
