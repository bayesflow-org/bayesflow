from collections.abc import Sequence, Mapping, Callable

import numpy as np

from ...utils.dict_utils import dicts_to_arrays, compute_test_quantities


def posterior_z_score(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    test_quantities: dict[str, Callable] = None,
    aggregation: Callable | None = np.median,
) -> dict[str, any]:
    """
    Computes the posterior z-score from prior to posterior for the given samples according to [1]:

    post_z_score = (posterior_mean - true_parameters) / posterior_std

    The score is adequate if it centers around zero and spreads roughly
    in the interval [-3, 3]

    [1] Schad, D. J., Betancourt, M., & Vasishth, S. (2021).
    Toward a principled Bayesian workflow in cognitive science.
    Psychological methods, 26(1), 103.

    Paper also available at https://arxiv.org/abs/1904.12765

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
            The (optionally aggregated) posterior z-score per variable
        - "metric_name" : str
            The name of the metric ("Posterior z-score").
        - "variable_names" : str
            The (inferred) variable names.

    Notes
    -----
    Posterior z-score quantifies how far the posterior mean lies from
    the true generating parameter, in standard-error units. Values near 0
    (in absolute terms) mean the posterior mean is close to the truth;
    large absolute values indicate substantial deviation.
    The sign shows the direction of the bias.

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
    post_means = samples["estimates"].mean(axis=1)
    post_stds = np.sqrt(post_vars)
    prior_vars = samples["targets"].var(axis=0, keepdims=True, ddof=1)
    z_score = (post_means - samples["targets"]) / post_stds
    if aggregation is not None:
        z_score = aggregation(z_score, axis=0)
    variable_names = samples["estimates"].variable_names
    return {"values": z_score, "metric_name": "Posterior z-score", "variable_names": variable_names}
