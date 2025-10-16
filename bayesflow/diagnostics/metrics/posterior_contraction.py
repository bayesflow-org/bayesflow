from collections.abc import Sequence, Mapping, Callable

import numpy as np

from ...utils.dict_utils import dicts_to_arrays, compute_test_quantities


def posterior_contraction(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    test_quantities: dict[str, Callable] = None,
    aggregation: Callable | None = np.median,
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
    test_quantities   : dict or None, optional, default: None
        A dict that maps plot titles to functions that compute
        test quantities based on estimate/target draws.

        The dict keys are automatically added to ``variable_keys``
        and ``variable_names``.
        Test quantity functions are expected to accept a dict of draws with
        shape ``(batch_size, ...)`` as the first (typically only)
        positional argument and return an NumPy array of shape
        ``(batch_size,)``.
        The functions do not have to deal with an additional
        sample dimension, as appropriate reshaping is done internally.
    aggregation    : callable or None, optional (default = np.median)
        Function to aggregate the PC across draws. Typically `np.mean` or `np.median`.
        If None is provided, the individual values are returned.

    Returns
    -------
    result : dict
        Dictionary containing:

        - "values" : float or np.ndarray
            The (optionally aggregated) posterior contraction per variable
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
        estimates=estimates,
        targets=targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
    )

    post_vars = samples["estimates"].var(axis=1, ddof=1)
    prior_vars = samples["targets"].var(axis=0, keepdims=True, ddof=1)
    contraction = np.clip(1 - (post_vars / prior_vars), 0, 1)
    if aggregation is not None:
        contraction = aggregation(contraction, axis=0)
    variable_names = samples["estimates"].variable_names
    return {"values": contraction, "metric_name": "Posterior Contraction", "variable_names": variable_names}
