from collections.abc import Callable, Mapping, Sequence

import inspect
import keras
from typing import TypeVar, Any

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


def split_arrays(data: Mapping[str, np.ndarray], axis: int = -1) -> Mapping[str, np.ndarray]:
    """Split tensors in the dictionary along the given axis."""
    result = {}

    for key, value in data.items():
        if len(value.shape) == 1:
            result[key] = value
            continue

        if value.shape[axis] == 1:
            result[key] = np.squeeze(value, axis=axis)
            continue

        splits = np.split(value, value.shape[axis], axis=axis)
        splits = [np.squeeze(split, axis=axis) for split in splits]

        for i, split in enumerate(splits):
            result[f"{key}_{i + 1}"] = split

    return result


def dicts_to_arrays(
    targets: Mapping[str, np.ndarray] | np.ndarray,
    references: Mapping[str, np.ndarray] | np.ndarray = None,
    variable_names: Sequence[str] = None,
    default_name: str = "var",
) -> Mapping[str, Any]:
    """Helper function that prepares estimates and optional ground truths for diagnostics
    (plotting or computation of metrics).

    The function operates on both arrays and dictionaries and assumes either a dictionary
    where each key contains a 1D or a 2D array (i.e., a univariate quantity or samples thereof)
    or a 2D or 3D array where the last axis represents all quantities of interest.

    If a `ground_truths` array is provided, it must correspond to estimates in terms of type
    and structure of the first and last axis.

    If a dictionary is provided, `variable_names` acts as a filter to select variables from
    estimates. If an array is provided, `variable_names` can be used to override the `default_name`.

    Parameters
    ----------
    targets   : dict[str, ndarray] or ndarray
        The model-generated predictions or estimates, which can take the following forms:
        - ndarray of shape (num_datasets, num_variables)
            Point estimates for each dataset, where `num_datasets` is the number of datasets
            and `num_variables` is the number of variables per dataset.
        - ndarray of shape (num_datasets, num_draws, num_variables)
            Posterior samples for each dataset, where `num_datasets` is the number of datasets,
            `num_draws` is the number of posterior draws, and `num_variables` is the number of variables.

    references : dict[str, ndarray] or ndarray, optional (default = None)
        Ground-truth values corresponding to the estimates. Must match the structure and dimensionality
        of `estimates` in terms of first and last axis.

    variable_names : Sequence[str], optional (default = None)
        Optional variable names to act as a filter if dicts provided or actual variable names in case of array
        inputs.
    default_name   : str, optional (default = "v")
        The default variable name to use if array arguments and no variable names are provided.
    """

    # Ensure that posterior and prior variables have the same type
    if references is not None:
        if type(targets) is not type(references):
            raise ValueError("You should either use dicts or tensors, but not separate types for your inputs.")

    # Case dictionaries provided
    if isinstance(targets, dict):
        targets = split_arrays(targets)
        variable_names = list(targets.keys()) if variable_names is None else variable_names
        targets = np.stack([v for k, v in targets.items() if k in variable_names], axis=-1)

        if references is not None:
            references = split_arrays(references)
            references = np.stack([v for k, v in references.items() if k in variable_names], axis=-1)

    # Case arrays provided
    elif isinstance(targets, np.ndarray):
        if variable_names is None:
            variable_names = [f"${default_name}_{{{i}}}$" for i in range(targets.shape[-1])]

    # Throw if unknown type
    else:
        raise TypeError(
            f"Only dicts and tensors are supported as arguments, " f"but your targets are of type {type(targets)}"
        )

    return dict(
        targets=targets,
        references=references,
        variable_names=variable_names,
        num_variables=len(variable_names),
    )
