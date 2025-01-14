from typing import Sequence, Any, Mapping, Callable

import numpy as np

from ...utils.dict_utils import dicts_to_arrays


def root_mean_squared_error(
    targets: Mapping[str, np.ndarray] | np.ndarray,
    references: Mapping[str, np.ndarray] | np.ndarray,
    normalize: bool = True,
    aggregation: Callable = np.median,
    variable_names: Sequence[str] = None,
) -> Mapping[str, Any]:
    """Computes the (Normalized) Root Mean Squared Error (RMSE/NRMSE) for the given posterior and prior samples.

    Parameters
    ----------
    targets   : np.ndarray of shape (num_datasets, num_draws_post, num_variables)
        Posterior samples, comprising `num_draws_post` random draws from the posterior distribution
        for each data set from `num_datasets`.
    references  : np.ndarray of shape (num_datasets, num_variables)
        Prior samples, comprising `num_datasets` ground truths.
    normalize      : bool, optional (default = True)
        Whether to normalize the RMSE using the range of the prior samples.
    aggregation    : callable, optional (default = np.median)
        Function to aggregate the RMSE across draws. Typically `np.mean` or `np.median`.
    variable_names : Sequence[str], optional (default = None)
        Optional variable names to select from the available variables.

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

    samples = dicts_to_arrays(targets=targets, references=references, variable_names=variable_names)

    rmse = np.sqrt(np.mean((samples["targets"] - samples["references"][:, None, :]) ** 2, axis=0))

    if normalize:
        rmse /= (samples["references"].max(axis=0) - samples["references"].min(axis=0))[None, :]
        metric_name = "NRMSE"
    else:
        metric_name = "RMSE"

    rmse = aggregation(rmse, axis=0)
    return {"values": rmse, "metric_name": metric_name, "variable_names": samples["variable_names"]}
