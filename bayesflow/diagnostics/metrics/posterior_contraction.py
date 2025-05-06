from collections.abc import Sequence, Mapping, Callable

import numpy as np

from ...utils.dict_utils import dicts_to_arrays


def posterior_contraction(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    aggregation: Callable = np.median,
) -> dict[str, any]:
    """
    Computes the posterior contraction (PC) from prior to posterior for the given samples.

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
    aggregation    : callable, optional (default = np.median)
        Function to aggregate the PC across draws. Typically `np.mean` or `np.median`.

    Returns
    -------
    result : dict
        Dictionary containing:

        - "values" : float or np.ndarray
            The aggregated posterior contraction per variable
        - "metric_name" : str
            The name of the metric ("Posterior Contraction").
        - "variable_names" : str
            The (inferred) variable names.

    Notes
    -----
    Posterior contraction measures the reduction in uncertainty from the prior to the posterior.
    Values close to 1 indicate strong contraction (high reduction in uncertainty), while values close to 0
    indicate low contraction.
    """

    samples = dicts_to_arrays(
        estimates=estimates,
        targets=targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
    )

    post_vars = samples["estimates"].var(axis=1, ddof=1)
    prior_vars = samples["targets"].var(axis=0, keepdims=True, ddof=1)
    contraction = np.clip(1 - (post_vars / prior_vars), 0, 1)
    contraction = aggregation(contraction, axis=0)
    variable_names = samples["estimates"].variable_names
    return {"values": contraction, "metric_name": "Posterior Contraction", "variable_names": variable_names}
