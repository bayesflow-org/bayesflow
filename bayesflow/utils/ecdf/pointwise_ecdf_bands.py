from collections.abc import Sequence
import numpy as np
from scipy.stats import binom as scipy_binomial


def pointwise_ecdf_bands(
    num_estimates: int,
    num_points: int = None,
    confidence: float = 0.99,
    min_point: float = 1e-5,
    max_point: float = 1 - 1e-5,
    max_num_points: int = 1000,
) -> Sequence:
    """Computes the pointwise ECDF confidence bands from the inverse CDF of the binomial distribution.

    Refer to the following for context and notation:

    Säilynoja, T., Bürkner, P. C., & Vehtari, A. (2022).
    Graphical test for discrete uniformity and its applications in goodness-of-fit
    evaluation and multiple sample comparison. Statistics and Computing, 32(2), 32.
    See: https://link.springer.com/article/10.1007/s11222-022-10090-6

    Will be used by the diagnostics module to create the ECDF marginal calibration plots.

    Parameters
    ----------
    num_estimates   : int
        The sample size used for computing the ECDF. Will equal to the number of simulated conditions when used
        for simulation-based calibration.
    num_points      : int, optional, default: None
        The number of evaluation points on the interval (0, 1). Defaults to `num_points = num_estimates` if
        not explicitly specified. Correspond to `K` in the paper above.
    confidence      : float in (0, 1), optional, default: 0.95
        The confidence level, `confidence = 1 - alpha` specifies the width of the confidence interval.
    min_point       : float, optional, default: 1e-5
        Lower bound of the interval for which the confidence bands are computed.
    max_point       : float, optional, default: 1-1e-5
        Upper bound of the interval for which the confidence bands are computed.
    max_num_points  : int, optional, default: 1000
        Upper bound on `num_points`. Saves computation time when `num_estimates` is large.

    Returns
    -------
    (alpha, z, L, U) - tuple of scalar and three arrays of size (num_estimates,) containing the confidence level
        as well as the evaluation points, the lower, and the upper confidence bands, respectively.
    """

    N = num_estimates
    if num_points is None:
        K = min(N, max_num_points)
    else:
        K = min(num_points, max_num_points)
    z = np.linspace(min_point, max_point, K)

    alpha = 1 - confidence
    L = np.zeros_like(z)
    U = np.zeros_like(z)

    for i, p in enumerate(z):
        L[i] = scipy_binomial.ppf(alpha / 2, N, p) / N
        U[i] = scipy_binomial.ppf(1 - alpha / 2, N, p) / N

    return alpha, z, L, U
