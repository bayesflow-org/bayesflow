import inspect
from collections.abc import Callable, Mapping, Sequence
from typing import TypeVar, Any

import keras
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


def filter_kwargs(kwargs: dict[str, T], f: Callable) -> dict[str, T]:
    """Filter keyword arguments for f"""
    signature = inspect.signature(f)

    for parameter in signature.parameters.values():
        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            # there is a **kwargs parameter, so anything is valid
            return kwargs

    kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}

    return kwargs


def layer_kwargs(kwargs: dict[str, T]) -> dict[str, T]:
    """Filter keyword arguments for keras.Layer"""
    valid_keys = ["dtype", "name", "trainable"]
    return {key: value for key, value in kwargs.items() if key in valid_keys}


def model_kwargs(kwargs: dict[str, T]) -> dict[str, T]:
    """Filter keyword arguments for keras.Model"""
    return layer_kwargs(kwargs)


def sequential_kwargs(kwargs: dict[str, T]) -> dict[str, T]:
    """Filter keyword arguments for keras.Sequential"""
    valid_keys = ["name", "trainable"]
    return {key: value for key, value in kwargs.items() if key in valid_keys}


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
        if not hasattr(value, "shape"):
            result[key] = np.array([value])
            continue

        if len(value.shape) == 1:
            result[key] = value
            continue

        if value.shape[axis] == 1:
            result[key] = np.squeeze(value, axis=axis)
            continue

        splits = np.split(value, value.shape[axis], axis=axis)
        splits = [np.squeeze(split, axis=axis) for split in splits]

        for i, split in enumerate(splits):
            result[f"{key}_{i}"] = split

    return result


class VariableArray(np.ndarray):
    """
    An enriched numpy array with information on variable keys and names
    to be used in post-processing, specifically the diagnostics module.

    The current implemention is very basic and we may want to extend it
    in the future should this general structure prove useful.

    Design according to
    https://numpy.org/doc/stable/user/basics.subclassing.html#simple-example-adding-an-extra-attribute-to-ndarray
    """

    def __new__(cls, input_array, variable_keys=None, variable_names=None):
        obj = np.asarray(input_array).view(cls)
        obj.variable_keys = variable_keys
        obj.variable_names = variable_names
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.variable_keys = getattr(obj, "variable_keys", None)
        self.variable_names = getattr(obj, "variable_names", None)


def make_variable_array(
    x: Mapping[str, np.ndarray] | np.ndarray,
    dataset_ids: Sequence[int] | int = None,
    variable_keys: Sequence[str] | str = None,
    variable_names: Sequence[str] | str = None,
    default_name: str = "v",
) -> VariableArray:
    """
    Helper function to validate arrays for use in the diagnostics module.

    Parameters
    ----------
    x   : dict[str, ndarray] or ndarray. Dict of arrays or array to be validated.
        See dicts_to_arrays
    dataset_ids : Sequence of integers indexing the datasets to select (default = None).
        By default, use all datasets.
    variable_names : Sequence[str], optional (default = None)
        Optional variable names to act as a filter if dicts provided or actual variable names in case of array
        inputs.
    default_name   : str, optional (default = "v")
        The default variable name to use if array arguments and no variable names are provided.
    """

    if isinstance(variable_keys, str):
        variable_keys = [variable_keys]

    if isinstance(variable_names, str):
        variable_names = [variable_names]

    if isinstance(x, dict):
        if variable_keys is not None:
            x = {k: x[k] for k in variable_keys}

        variable_keys = x.keys()

        if dataset_ids is not None:
            if isinstance(dataset_ids, int):
                # dataset_ids needs to be a sequence so that np.stack works correctly
                dataset_ids = [dataset_ids]

            x = {k: v[dataset_ids] for k, v in x.items()}

        x = split_arrays(x)

        if variable_names is None:
            variable_names = list(x.keys())

        x = np.stack(list(x.values()), axis=-1)

    # Case arrays provided
    elif isinstance(x, np.ndarray):
        if isinstance(x, VariableArray):
            # reuse existing variable keys and names if contained in x
            if variable_names is None:
                variable_names = x.variable_names
            if variable_keys in None:
                variable_keys = x.variable_keys

        # use default names if not otherwise specified
        if variable_names is None:
            variable_names = [f"{default_name}_{i}" for i in range(x.shape[-1])]

        if dataset_ids is not None:
            x = x[dataset_ids]

    # Throw if unknown type
    else:
        raise TypeError(f"Only dicts and tensors are supported as arguments, but your estimates are of type {type(x)}")

    if len(variable_names) != x.shape[-1]:
        raise ValueError("Length of 'variable_names' should be the same as the number of variables.")

    if variable_keys is None:
        # every variable will count as its own key if not otherwise specified
        variable_keys = variable_names

    x = VariableArray(x, variable_keys=variable_keys, variable_names=variable_names)

    return x


def dicts_to_arrays(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray = None,
    priors: Mapping[str, np.ndarray] | np.ndarray = None,
    dataset_ids: Sequence[int] | int = None,
    variable_keys: Sequence[str] | str = None,
    variable_names: Sequence[str] | str = None,
    default_name: str = "v",
) -> dict[str, Any]:
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
    estimates   : dict[str, ndarray] or ndarray
        The model-generated predictions or estimates, which can take the following forms:
        - ndarray of shape (num_datasets, num_variables)
            Point estimates for each dataset, where `num_datasets` is the number of datasets
            and `num_variables` is the number of variables per dataset.
        - ndarray of shape (num_datasets, num_draws, num_variables)
            Posterior samples for each dataset, where `num_datasets` is the number of datasets,
            `num_draws` is the number of posterior draws, and `num_variables` is the number of variables.

    targets : dict[str, ndarray] or ndarray, optional (default = None)
        Ground-truth values corresponding to the estimates. Must match the structure and dimensionality
        of `estimates` in terms of first and last axis.

    dataset_ids : Sequence of integers indexing the datasets to select (default = None).
        By default, use all datasets.

    variable_keys : list or None, optional, default: None
       Select keys from the dictionary provided in samples.
       By default, select all keys.

    variable_names : Sequence[str], optional (default = None)
        Optional variable names to act as a filter if dicts provided or actual variable names in case of array
        inputs.

    default_name   : str, optional (default = "v")
        The default variable name to use if array arguments and no variable names are provided.
    """

    # other to be validated arrays (see below) will take use
    # the variable_keys and variable_names implied by estimates
    estimates = make_variable_array(
        estimates,
        dataset_ids=dataset_ids,
        variable_keys=variable_keys,
        variable_names=variable_names,
        default_name=default_name,
    )

    if targets is not None:
        targets = make_variable_array(
            targets,
            dataset_ids=dataset_ids,
            variable_keys=estimates.variable_keys,
            variable_names=estimates.variable_names,
        )

    if priors is not None:
        priors = make_variable_array(
            priors,
            # priors are data independent so datasets_ids is not passed here
            variable_keys=estimates.variable_keys,
            variable_names=estimates.variable_names,
        )

    return dict(
        estimates=estimates,
        targets=targets,
        priors=priors,
    )


def squeeze_inner_estimates_dict(estimates):
    """If a dictionary has only one key-value pair and the key is "value", return only its value.
    Otherwise, return the unchanged dictionary.

    This method helps to remove unnecessary nesting levels.
    """
    if len(estimates.keys()) == 1 and "value" in estimates.keys():
        return estimates["value"]
    else:
        return estimates
