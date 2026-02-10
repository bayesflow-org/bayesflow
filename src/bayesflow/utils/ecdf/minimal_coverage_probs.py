import numpy as np
from scipy.stats import binom as scipy_binomial


def minimal_coverage_probs(z: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Vectorized function to compute the minimal coverage probability for uniform
    empirical cumulative distribution functions (ECDFs) given evaluation points z
    and a sample of uniform draws.

    See: https://link.springer.com/article/10.1007/s11222-022-10090-6
    Säilynoja, T., Bürkner, P. C., & Vehtari, A. (2022).
    Graphical test for discrete uniformity and its applications in goodness-of-fit
    evaluation and multiple sample comparison. Statistics and Computing, 32(2), 32.

    Parameters
    ----------
    z  : np.ndarray of shape (num_points, )
        The vector of evaluation points.
    u  : np.ndarray of shape (num_simulations, num_samples)
        The matrix of simulated draws (samples) from U(0, 1)
    """

    N = u.shape[1]
    F_m = np.sum((z[:, np.newaxis] >= u[:, np.newaxis, :]), axis=-1) / u.shape[1]
    bin1 = scipy_binomial(N, z).cdf(N * F_m)
    bin2 = scipy_binomial(N, z).cdf(N * F_m - 1)
    gamma = 2 * np.min(np.min(np.stack([bin1, 1 - bin2], axis=-1), axis=-1), axis=-1)
    return gamma
