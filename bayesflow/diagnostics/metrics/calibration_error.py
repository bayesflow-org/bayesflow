from typing import Sequence, Any

import numpy as np

from ...utils.dict_utils import dicts_to_arrays


def calibration_error(
    post_samples: dict[str, np.ndarray] | np.ndarray,
    prior_samples: dict[str, np.ndarray] | np.ndarray,
    resolution: int = 20,
    aggregation: callable = np.median,
    min_quantile: float = 0.005,
    max_quantile: float = 0.995,
    filter_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
) -> dict[str, Any]:
    """Computes an aggregate score for the marginal calibration error over an ensemble of approximate
    posteriors. The calibration error is given as the aggregate (e.g., median) of the absolute deviation
    between an alpha-CI and the relative number of inliers from ``prior_samples`` over multiple alphas in
    (0, 1).

    Parameters
    ----------
    post_samples  : np.ndarray of shape (num_datasets, num_draws, num_variables)
        The random draws from the approximate posteriors over ``num_datasets``
    prior_samples : np.ndarray of shape (num_datasets, num_variables)
        The corresponding ground-truth values sampled from the prior
    resolution    : int, optional, default: 20
        The number of credibility intervals (CIs) to consider
    aggregation   : callable or None, optional, default: np.median
        The function used to aggregate the marginal calibration errors.
        If ``None`` provided, the per-alpha calibration errors will be returned.
    min_quantile  : float in (0, 1), optional, default: 0.005
        The minimum posterior quantile to consider.
    max_quantile  : float in (0, 1), optional, default: 0.995
        The maximum posterior quantile to consider.
    filter_keys    : Sequence[str], optional (default = None)
        Optional variable names to filter out of the metric computation.
    variable_names : Sequence[str], optional (default = None)
        Optional variable names to select from the available variables.

    Returns
    -------
    result : dict
        Dictionary containing:
        - "metric" : float or np.ndarray
            The aggregated calibration error per variable
        - "metric_name" : str
            The name of the metric ("Calibration Error").
        - "variable_names" : str
            The (inferred) variable names.
    """

    samples = dicts_to_arrays(post_samples, prior_samples, filter_keys, variable_names)

    # Define alpha values and the corresponding quantile bounds
    alphas = np.linspace(start=min_quantile, stop=max_quantile, num=resolution)
    regions = 1 - alphas
    lowers = regions / 2
    uppers = 1 - lowers

    # Compute quantiles for each alpha, for each dataset and parameter
    quantiles = np.quantile(samples["post_variables"], [lowers, uppers], axis=1)

    # Shape: (2, resolution, num_datasets, num_params)
    lower_bounds, upper_bounds = quantiles[0], quantiles[1]

    # Compute masks for inliers
    lower_mask = lower_bounds <= samples["prior_variables"][None, ...]
    upper_mask = upper_bounds >= samples["prior_variables"][None, ...]

    # Logical AND to identify inliers for each alpha
    inlier_id = np.logical_and(lower_mask, upper_mask)

    # Compute the relative number of inliers for each alpha
    alpha_pred = np.mean(inlier_id, axis=1)

    # Calculate absolute error between predicted inliers and alpha
    absolute_errors = np.abs(alpha_pred - alphas[:, None])

    # Aggregate errors across alpha
    error = aggregation(absolute_errors, axis=0)

    return {"metric": error, "metric_name": "Calibration Error", "variable_names": variable_names}
