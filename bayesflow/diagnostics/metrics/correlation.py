from typing import Any
from collections.abc import Sequence, Mapping, Callable

import numpy as np

from ...utils.dict_utils import dicts_to_arrays, compute_test_quantities


def correlation(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    test_quantities: dict[str, Callable] = None,
    aggregation: Callable = np.median,
) -> dict[str, Any]:
    """
    Computes the Pearson correlation between estimates and targets for each random
    draw from the posterior distribution across datasets, separately for each variable.

    Parameters
    ----------
    estimates   : np.ndarray or dict[str, np.ndarray]
        Posterior samples, either as a NumPy array of shape (num_datasets, num_draws_post, num_variables)
        or as a dictionary mapping variable names to arrays.
        Comprises `num_draws_post` random draws from the posterior distribution
        for each data set from `num_datasets`.
    targets  : np.ndarray or dict[str, np.ndarray]
        Prior samples, either as a NumPy array of shape (num_datasets, num_variables)
        or as a dictionary mapping variable names to arrays.
        Comprises `num_datasets` ground truths.
    variable_keys : Sequence[str], optional (default = None)
       Select keys from the dictionaries provided in estimates and targets.
       By default, select all keys.
    variable_names : Sequence[str], optional (default = None)
        Optional variable names to show in the output.
    test_quantities   : dict or None, optional, default: None
        A dict that maps plot titles to functions that compute
        test quantities based on estimate/target draws.

        The dict keys are automatically added to ``variable_keys``
        and ``variable_names``.
        Test quantity functions are expected to accept a dict of draws with
        shape ``(batch_size, ...)`` as the first positional argument and return
        a NumPy array of shape ``(batch_size,)``.
    aggregation    : callable, optional (default = np.median)
        Function to aggregate the correlations across posterior draws.
        Typically `np.mean` or `np.median`.

    Returns
    -------
    result : dict
        Dictionary containing:

        - "values" : np.ndarray
            The aggregated Pearson correlation for each variable.
        - "metric_name" : str
            The name of the metric ("Correlation").
        - "variable_names" : str
            The (inferred) variable names.
    """

    # Optionally, compute and prepend test quantities from draws
    if test_quantities is not None:
        updated_data = compute_test_quantities(
            targets=targets,
            estimates=estimates,
            variable_keys=variable_keys,
            variable_names=variable_names,
            test_quantities=test_quantities,
        )
        variable_names = updated_data["variable_names"]
        variable_keys = updated_data["variable_keys"]
        estimates = updated_data["estimates"]
        targets = updated_data["targets"]

    samples = dicts_to_arrays(
        estimates=estimates, targets=targets, variable_keys=variable_keys, variable_names=variable_names
    )

    est = np.asarray(samples["estimates"])
    tgt = np.asarray(samples["targets"])

    # Correlation across datasets, separately for each posterior draw and variable
    est_centered = est - np.mean(est, axis=0, keepdims=True)
    tgt_centered = tgt[:, None, :] - np.mean(tgt, axis=0, keepdims=True)[:, None, :]

    numerator = np.sum(est_centered * tgt_centered, axis=0)
    denominator = np.sqrt(np.sum(est_centered**2, axis=0) * np.sum(tgt_centered**2, axis=0))

    corr = np.divide(
        numerator,
        denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )

    corr = aggregation(corr, axis=0)

    variable_names = samples["estimates"].variable_names
    return {
        "values": corr,
        "metric_name": "Correlation",
        "variable_names": variable_names,
    }
