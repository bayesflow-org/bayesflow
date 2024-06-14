
import keras

from keras import ops
from scipy.stats import binom
from sklearn.calibration import calibration_curve


def expected_calibration_error(
    m_true,
    m_pred,
    num_bins: int = 10
):
    """Estimates the expected calibration error (ECE) of a model comparison network according to [1].

    [1] Naeini, M. P., Cooper, G., & Hauskrecht, M. (2015).
        Obtaining well calibrated probabilities using bayesian binning.
        In Proceedings of the AAAI conference on artificial intelligence (Vol. 29, No. 1).

    Notes
    -----
    Make sure that ``m_true`` are **one-hot encoded** classes!

    Parameters
    ----------
    m_true      : np.ndarray of shape (num_sim, num_models)
        The one-hot-encoded true model indices.
    m_pred      : tf.tensor of shape (num_sim, num_models)
        The predicted posterior model probabilities.
    num_bins    : int, optional, default: 10
        The number of bins to use for the calibration curves (and marginal histograms).

    Returns
    -------
    cal_errs    : list of length (num_models)
        The ECEs for each model.
    probs       : list of length (num_models)
        The bin information for constructing the calibration curves.
        Each list contains two arrays of length (num_bins) with the predicted and true probabilities for each bin.
    """

    # Convert tf.Tensors to numpy, if passed
    if type(m_true) is not np.ndarray:
        m_true = m_true.numpy()
    if type(m_pred) is not np.ndarray:
        m_pred = m_pred.numpy()

    # Extract number of models and prepare containers
    n_models = m_true.shape[1]
    cal_errs = []
    probs_true = []
    probs_pred = []

    # Loop for each model and compute calibration errs per bin
    for k in range(n_models):
        y_true = (m_true.argmax(axis=1) == k).astype(float32)
        y_prob = m_pred[:, k]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=num_bins)

        # Compute ECE by weighting bin errors by bin size
        bins = ops.linspace(0.0, 1.0, num_bins + 1)
        binids = ops.searchsorted(bins[1:-1], y_prob)
        bin_total = ops.bincount(binids, minlength=len(bins))
        nonzero = bin_total != 0
        cal_err = ops.sum(ops.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))

        cal_errs.append(cal_err)
        probs_true.append(prob_true)
        probs_pred.append(prob_pred)
    return cal_errs, probs_true, probs_pred


def get_coverage_probs(z, u):
    """Vectorized function to compute the minimal coverage probability for uniform
    ECDFs given evaluation points z and a sample of samples u.

    Parameters
    ----------
    z  : np.ndarray of shape (num_points, )
        The vector of evaluation points.
    u  : np.ndarray of shape (num_simulations, num_samples)
        The matrix of simulated draws (samples) from U(0, 1)
    """

    N = u.shape[1]
    F_m = ops.sum((z[:, np.newaxis] >= u[:, np.newaxis, :]), axis=-1) / u.shape[1]
    bin1 = binom(N, z).cdf(N * F_m)
    bin2 = binom(N, z).cdf(N * F_m - 1)
    gamma = 2 * ops.min(ops.min(ops.stack([bin1, 1 - bin2], axis=-1), axis=-1), axis=-1)
    return gamma


def simultaneous_ecdf_bands(
    num_samples: int,
    num_points: int = None,
    num_simulations: int = 1000,
    confidence: float = 0.95,
    eps: float = 1e-5,
    max_num_points: int = 1000
):
    """Computes the simultaneous ECDF bands through simulation according to
    the algorithm described in Section 2.2:

    https://link.springer.com/content/pdf/10.1007/s11222-022-10090-6.pdf

    Depends on the vectorized utility function `get_coverage_probs(z, u)`.

    Parameters
    ----------
    num_samples     : int
        The sample size used for computing the ECDF. Will equal to the number of posterior
        samples when used for calibrarion. Corresponds to `N` in the paper above.
    num_points      : int, optional, default: None
        The number of evaluation points on the interval (0, 1). Defaults to `num_points = num_samples` if
        not explicitly specified. Correspond to `K` in the paper above.
    num_simulations : int, optional, default: 1000
        The number of samples of size `n_samples` to simulate for determining the simultaneous CIs.
    confidence      : float in (0, 1), optional, default: 0.95
        The confidence level, `confidence = 1 - alpha` specifies the width of the confidence interval.
    eps             : float, optional, default: 1e-5
        Small number to add to the lower and subtract from the upper bound of the interval [0, 1]
        to avoid edge artefacts. No need to touch this.
    max_num_points  : int, optional, default: 1000
        Upper bound on `num_points`. Saves computation time when `num_samples` is large.

    Returns
    -------
    (alpha, z, L, U) - tuple of scalar and three arrays of size (num_samples,) containing the confidence level as well as
                       the evaluation points, the lower, and the upper confidence bands, respectively.
    """

    # Use shorter var names throughout
    N = num_samples
    if num_points is None:
        K = min(N, max_num_points)
    else:
        K = min(num_points, max_num_points)
    M = num_simulations

    # Specify evaluation points
    z = ops.linspace(0 + eps, 1 - eps, K)

    # Simulate M samples of size N
    u = keras.random.uniform(size=(M, N))

    # Get alpha
    alpha = 1 - confidence

    # Compute minimal coverage probabilities
    gammas = get_coverage_probs(z, u)

    # Use insights from paper to compute lower and upper confidence interval
    gamma = np.percentile(gammas, 100 * alpha)
    L = binom(N, z).ppf(gamma / 2) / N
    U = binom(N, z).ppf(1 - gamma / 2) / N
    return alpha, z, L, U