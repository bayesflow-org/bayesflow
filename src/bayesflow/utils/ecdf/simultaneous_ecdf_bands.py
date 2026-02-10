from collections.abc import Sequence
import numpy as np
from scipy.stats import binom as scipy_binomial

from .minimal_coverage_probs import minimal_coverage_probs


def simultaneous_ecdf_bands(
    num_estimates: int,
    num_points: int = None,
    num_simulations: int = 1000,
    confidence: float = 0.95,
    eps: float = 1e-5,
    max_num_points: int = 1000,
) -> Sequence:
    """Computes the simultaneous ECDF bands through simulation according to
    the algorithm described in Section 2.2 of

    Säilynoja, T., Bürkner, P. C., & Vehtari, A. (2022).
    Graphical test for discrete uniformity and its applications in goodness-of-fit
    evaluation and multiple sample comparison. Statistics and Computing, 32(2), 32.
    See: https://link.springer.com/article/10.1007/s11222-022-10090-6

    Depends on the vectorized utility function `ecdf.minimal_coverage_probs(z, u)`.
    Will be used by the diagnostics module to create the ECDF marginal calibration plots.

    Parameters
    ----------
    num_estimates   : int
        The sample size used for computing the ECDF. Will equal to the number of simulated conditions when used
        for simulation-based calibration. Corresponds to `N` in the paper above.
    num_points      : int, optional, default: None
        The number of evaluation points on the interval (0, 1). Defaults to `num_points = num_estimates` if
        not explicitly specified. Correspond to `K` in the paper above.
    num_simulations : int, optional, default: 1000
        The number of samples of size `n_samples` to simulate for determining the simultaneous CIs.
    confidence      : float in (0, 1), optional, default: 0.95
        The confidence level, `confidence = 1 - alpha` specifies the width of the confidence interval.
    eps             : float, optional, default: 1e-5
        Small number to add to the lower and subtract from the upper bound of the interval [0, 1]
        to avoid edge artefacts. No need to touch this.
    max_num_points  : int, optional, default: 1000
        Upper bound on `num_points`. Saves computation time when `num_estimates` is large.

    Returns
    -------
    (alpha, z, L, U) - tuple of scalar and three arrays of size (num_estimates,) containing the confidence level
        as well as the evaluation points, the lower, and the upper confidence bands, respectively.
    """

    # Use shorter variable names to match paper notation
    N = num_estimates
    if num_points is None:
        K = min(N, max_num_points)
    else:
        K = min(num_points, max_num_points)
    M = num_simulations

    # Specify evaluation points
    z = np.linspace(0 + eps, 1 - eps, K)

    # Simulate M samples of size N
    u = np.random.uniform(size=(M, N))

    # Compute minimal coverage probabilities
    gammas = minimal_coverage_probs(z, u)

    # Use insights from paper to compute lower and upper confidence interval
    alpha = 1 - confidence
    gamma = np.percentile(gammas, 100 * alpha)
    lower_ci = scipy_binomial(N, z).ppf(gamma / 2) / N
    upper_ci = scipy_binomial(N, z).ppf(1 - gamma / 2) / N
    return alpha, z, lower_ci, upper_ci
