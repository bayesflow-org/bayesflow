from typing import Sequence, Any, Mapping, Callable

import numpy as np

from ...utils.dict_utils import dicts_to_arrays


def posterior_contraction(
    targets: Mapping[str, np.ndarray] | np.ndarray,
    references: Mapping[str, np.ndarray] | np.ndarray,
    aggregation: Callable = np.median,
    variable_names: Sequence[str] = None,
) -> Mapping[str, Any]:
    """Computes the posterior contraction (PC) from prior to posterior for the given samples.

    Parameters
    ----------
    targets   : np.ndarray of shape (num_datasets, num_draws_post, num_variables)
        Posterior samples, comprising `num_draws_post` random draws from the posterior distribution
        for each data set from `num_datasets`.
    references  : np.ndarray of shape (num_datasets, num_variables)
        Prior samples, comprising `num_datasets` ground truths.
    aggregation    : callable, optional (default = np.median)
        Function to aggregate the PC across draws. Typically `np.mean` or `np.median`.
    variable_names : Sequence[str], optional (default = None)
        Optional variable names to select from the available variables.

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

    samples = dicts_to_arrays(targets=targets, references=references, variable_names=variable_names)

    post_vars = samples["targets"].var(axis=1, ddof=1)
    prior_vars = samples["references"].var(axis=0, keepdims=True, ddof=1)
    contraction = 1 - (post_vars / prior_vars)
    contraction = aggregation(contraction, axis=0)
    return {"values": contraction, "metric_name": "Posterior Contraction", "variable_names": samples["variable_names"]}
