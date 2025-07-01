from collections.abc import Mapping, Sequence

import numpy as np
from scipy.stats import binom

from ...utils.dict_utils import dicts_to_arrays


def calibration_log_gamma(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    num_null_draws: int = 1000,
    quantile: float = 0.05,
):
    """
    Compute the log gamma discrepancy statistic to test posterior calibration,
    see [1] for additional information.
    Log gamma is log(gamma/gamma_null), where gamma_null is the 5th percentile of the
    null distribution under uniformity of ranks.
    That is, if adopting a hypothesis testing framework,then log_gamma < 0 implies
    a rejection of the hypothesis of uniform ranks at the 5% level.
    This diagnostic is typically more sensitive than the Kolmogorov-Smirnoff test or
    ChiSq test.

    [1]  Martin Modrák. Angie H. Moon. Shinyoung Kim. Paul Bürkner. Niko Huurre.
    Kateřina Faltejsková. Andrew Gelman. Aki Vehtari.
    "Simulation-Based Calibration Checking for Bayesian Computation:
    The Choice of Test Quantities Shapes Sensitivity."
    Bayesian Anal. 20 (2) 461 - 488, June 2025. https://doi.org/10.1214/23-BA1404

    Parameters
    ----------
    estimates  : np.ndarray of shape (num_datasets, num_draws, num_variables)
        The random draws from the approximate posteriors over ``num_datasets``
    targets : np.ndarray of shape (num_datasets, num_variables)
        The corresponding ground-truth values sampled from the prior
    variable_keys : Sequence[str], optional (default = None)
       Select keys from the dictionaries provided in estimates and targets.
       By default, select all keys.
    variable_names : Sequence[str], optional (default = None)
        Optional variable names to show in the output.
    quantile : float in (0, 1), optional, default 0.05
        The quantile from the null distribution to be used as a threshold.
        A lower quantile increases sensitivity to deviations from uniformity.

    Returns
    -------
    result : dict
        Dictionary containing:

        - "values" : float or np.ndarray
            The log gamma values per variable
        - "metric_name" : str
            The name of the metric ("Log Gamma").
        - "variable_names" : str
            The (inferred) variable names.
    """
    samples = dicts_to_arrays(
        estimates=estimates,
        targets=targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
    )

    num_ranks = samples["estimates"].shape[0]
    num_post_draws = samples["estimates"].shape[1]

    # rank statistics
    ranks = np.sum(samples["estimates"] < samples["targets"][:, None], axis=1)

    # null distribution and threshold
    null_distribution = gamma_null_distribution(num_ranks, num_post_draws, num_null_draws)
    null_quantile = np.quantile(null_distribution, quantile)

    # compute log gamma for each parameter
    log_gammas = np.empty(ranks.shape[-1])

    for i in range(ranks.shape[-1]):
        gamma = gamma_discrepancy(ranks[:, i], num_post_draws=num_post_draws)
        log_gammas[i] = np.log(gamma / null_quantile)

    output = {
        "values": log_gammas,
        "metric_name": "Log Gamma",
        "variable_names": samples["estimates"].variable_names,
    }

    return output


def gamma_null_distribution(num_ranks: int, num_post_draws: int = 1000, num_null_draws: int = 1000) -> np.ndarray:
    """
    Computes the distribution of expected gamma values under uniformity of ranks.

    Parameters
    ----------
    num_ranks : int
        Number of ranks to use for each gamma.
    num_post_draws : int, optional, default 1000
        Number of posterior draws that were used to calculate the rank distribution.
    num_null_draws : int, optional, default 1000
        Number of returned gamma values under uniformity of ranks.

    Returns
    -------
    result : np.ndarray
        Array of shape (num_null_draws,) containing gamma values under uniformity of ranks.
    """
    z_i = np.arange(1, num_post_draws + 2) / (num_post_draws + 1)
    gamma = np.empty(num_null_draws)

    # loop non-vectorized to reduce memory footprint
    for i in range(num_null_draws):
        u = np.random.uniform(size=num_ranks)
        F_z = np.mean(u[:, None] < z_i, axis=0)
        bin_1 = binom.cdf(num_ranks * F_z, num_ranks, z_i)
        bin_2 = 1 - binom.cdf(num_ranks * F_z - 1, num_ranks, z_i)

        gamma[i] = 2 * np.min(np.minimum(bin_1, bin_2))

    return gamma


def gamma_discrepancy(ranks: np.ndarray, num_post_draws: int = 100) -> float:
    """
    Quantifies deviation from uniformity by the likelihood of observing the
    most extreme point on the empirical CDF of the given rank distribution
    according to [1] (equation 7).

    [1]  Martin Modrák. Angie H. Moon. Shinyoung Kim. Paul Bürkner. Niko Huurre.
    Kateřina Faltejsková. Andrew Gelman. Aki Vehtari.
    "Simulation-Based Calibration Checking for Bayesian Computation:
    The Choice of Test Quantities Shapes Sensitivity."
    Bayesian Anal. 20 (2) 461 - 488, June 2025. https://doi.org/10.1214/23-BA1404

    Parameters
    ----------
    ranks : array of shape (num_ranks,)
        Empirical rank distribution
    num_post_draws : int, optional, default 100
        Number of posterior draws used to generate ranks.

    Returns
    -------
    result : float
        Gamma discrepancy values for each parameter.
    """
    num_ranks = len(ranks)

    # observed count of ranks smaller than i
    R_i = np.array([sum(ranks < i) for i in range(1, num_post_draws + 2)])

    # expected proportion of ranks smaller than i
    z_i = np.arange(1, num_post_draws + 2) / (num_post_draws + 1)

    bin_1 = binom.cdf(R_i, num_ranks, z_i)
    bin_2 = 1 - binom.cdf(R_i - 1, num_ranks, z_i)

    # likelihood of obtaining the most extreme point on the empirical CDF
    # if the rank distribution was indeed uniform
    return float(2 * np.min(np.minimum(bin_1, bin_2)))
