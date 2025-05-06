from collections.abc import Sequence, Mapping, Callable

import numpy as np

from ...utils.dict_utils import dicts_to_arrays


def root_mean_squared_error(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    normalize: bool = True,
    aggregation: Callable = np.median,
) -> dict[str, any]:
    """
    Computes the (Normalized) Root Mean Squared Error (RMSE/NRMSE) for the given posterior and prior samples.

    Parameters
    ----------
    estimates   : np.ndarray of shape (num_datasets, num_draws_post, num_variables)
        Posterior samples, comprising `num_draws_post` random draws from the posterior distribution
        for each data set from `num_datasets`.
    targets  : np.ndarray of shape (num_datasets, num_variables)
        Prior samples, comprising `num_datasets` ground truths.
    variable_keys : Sequence[str], optional (default = None)
       Select keys from the dictionaries provided in estimates and targets.
       By default, select all keys.
    variable_names : Sequence[str], optional (default = None)
        Optional variable names to show in the output.
    normalize      : bool, optional (default = True)
        Whether to normalize the RMSE using the range of the prior samples.
    aggregation    : callable, optional (default = np.median)
        Function to aggregate the RMSE across draws. Typically `np.mean` or `np.median`.

    Notes
    -----
    Aggregation is performed after computing the RMSE for each posterior draw, instead of first aggregating
    the posterior draws and then computing the RMSE between aggregates and ground truths.

    Returns
    -------
    result : dict
        Dictionary containing:

        - "values" : np.ndarray
            The aggregated (N)RMSE for each variable.
        - "metric_name" : str
            The name of the metric ("RMSE" or "NRMSE").
        - "variable_names" : str
            The (inferred) variable names.
    """

    samples = dicts_to_arrays(
        estimates=estimates,
        targets=targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
    )

    rmse = np.sqrt(np.mean((samples["estimates"] - samples["targets"][:, None, :]) ** 2, axis=0))

    if normalize:
        rmse /= (samples["targets"].max(axis=0) - samples["targets"].min(axis=0))[None, :]
        metric_name = "NRMSE"
    else:
        metric_name = "RMSE"

    rmse = aggregation(rmse, axis=0)
    variable_names = samples["estimates"].variable_names
    return {"values": rmse, "metric_name": metric_name, "variable_names": variable_names}
