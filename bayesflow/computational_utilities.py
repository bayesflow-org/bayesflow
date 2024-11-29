# Copyright (c) 2022 The BayesFlow Developers

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import tensorflow as tf
from scipy import stats
from sklearn.calibration import calibration_curve
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPClassifier

from bayesflow.default_settings import MMD_BANDWIDTH_LIST
from bayesflow.exceptions import ShapeError


def posterior_calibration_error(
    posterior_samples,
    prior_samples,
    alpha_resolution=20,
    aggregator_fun=np.median,
    min_quantile=0.005,
    max_quantile=0.995,
):
    """Computes an aggregate score for the marginal calibration error over an ensemble of approximate
    posteriors. The calibration error is given as the aggregate (e.g., median) of the absolute deviation
    between an alpha-CI and the relative number of inliers from ``prior_samples`` over multiple alphas in
    (0, 1).

    Note: The function will use posterior quantiles for determining the credibility intervals. An alternative
    definition of the calibration error is possible using highest density intervals (HDIs).

    Parameters
    ----------
    posterior_samples  : np.ndarray of shape (num_datasets, num_draws, num_params)
        The random draws from the approximate posteriors over ``num_datasets``
    prior_samples      : np.ndarray of shape (num_datasets, num_params)
        The corresponding ground-truth values sampled from the prior
    alpha_resolution   : int, optional, default: 20
        The number of credibility intervals (CIs) to consider
    aggregator_fun     : callable or None, optional, default: np.median
        The function used to aggregate the marginal calibration errors.
        If ``None`` provided, the per-alpha calibration errors will be returned.
    min_quantile     : float in (0, 1), optional, default: 0.005
        The minimum posterior quantile to consider
    max_quantile     : float in (0, 1), optional, default: 0.995
        The maximum posterior quantile to consider

    Returns
    -------
    calibration_errors : np.ndarray of shape (num_params, ) or (alpha_resolution, num_params),
        if ``aggregator_fun is None``.
        The aggregated calibration error per marginal posterior.
    """

    num_params = prior_samples.shape[1]
    alphas = np.linspace(min_quantile, max_quantile, alpha_resolution)
    absolute_errors = np.zeros((alpha_resolution, num_params))

    for i, alpha in enumerate(alphas):
        # Find lower and upper bounds of posterior distribution
        region = 1 - alpha
        lower = region / 2
        upper = 1 - (region / 2)

        # Compute percentiles for given alpha using the entire posterior sample
        quantiles = np.quantile(posterior_samples, [lower, upper], axis=1)

        # Compute the relative number of inliers
        higher_mask = quantiles[0] <= prior_samples
        lower_mask = prior_samples <= quantiles[1]
        inlier_id = np.logical_and(higher_mask, lower_mask)
        alpha_pred = np.mean(inlier_id, axis=0)
        absolute_errors[i] = np.abs(alpha_pred - alpha)

    if aggregator_fun is not None:
        calibration_errors = aggregator_fun(absolute_errors, axis=0)
        return calibration_errors
    return absolute_errors


def compute_jacobian_trace(function, inputs, **kwargs):
    """Computes the exact Jacobian Trace of function with respect to inputs. Suitable for low dimensions (<32)

    Parameters
    ----------
    function : callable
        The function whose Jacobian at inputs will be computed
    inputs   : tf.Tensor of shape (batch_size, ...)
        The tensor with respect to which we are computing the Jacobian

    Returns
    -------
    outputs  : tf.Tensor of shape (batch_size, ...)
        The outputs of ``function``
    trace    : tf.Tensor of shape (batch_size, ...)
        The trace of the Jacobian at inputs.
    """

    if len(inputs.shape) == 2:
        batch_size, dims = inputs.shape
        trace = tf.zeros((batch_size,))
    else:
        batch_size, num_reps, dims = inputs.shape
        trace = tf.zeros((batch_size, num_reps))

    with tf.GradientTape(persistent=True) as tape:
        tape.watch(inputs)
        outputs = function(inputs, **kwargs)

    for step in range(dims):
        dummy = tf.cast(step * tf.ones(trace.shape), tf.int32)
        epsilon = tf.one_hot(dummy, dims)
        vjp = tape.gradient(outputs, inputs, output_gradients=epsilon)
        trace = trace + tf.reduce_sum(vjp * epsilon, axis=-1)
    return outputs, trace


def gaussian_kernel_matrix(x, y, sigmas=None):
    """Computes a Gaussian radial basis functions (RBFs) between the samples of x and y.

    We create a sum of multiple Gaussian kernels each having a width :math:`\sigma_i`.

    Parameters
    ----------
    x       :  tf.Tensor of shape (num_draws_x, num_features)
        Comprises `num_draws_x` Random draws from the "source" distribution `P`.
    y       :  tf.Tensor of shape (num_draws_y, num_features)
        Comprises `num_draws_y` Random draws from the "source" distribution `Q`.
    sigmas  : list(float), optional, default: None
        List which denotes the widths of each of the gaussians in the kernel.
        If `sigmas is None`, a default range will be used, contained in ``bayesflow.default_settings.MMD_BANDWIDTH_LIST``

    Returns
    -------
    kernel  : tf.Tensor of shape (num_draws_x, num_draws_y)
        The kernel matrix between pairs from `x` and `y`.
    """

    if sigmas is None:
        sigmas = MMD_BANDWIDTH_LIST
    norm = lambda v: tf.reduce_sum(tf.square(v), 1)
    beta = 1.0 / (2.0 * (tf.expand_dims(sigmas, 1)))
    dist = tf.transpose(norm(tf.expand_dims(x, 2) - tf.transpose(y)))
    s = tf.matmul(beta, tf.reshape(dist, (1, -1)))
    kernel = tf.reshape(tf.reduce_sum(tf.exp(-s), 0), tf.shape(dist))
    return kernel

def energy_kernel_matrix(x, y):
    """Computes an energy kernel between the samples of x and y as suggested
    in [1,2].

    [1] Feydy, J., Séjourné, T., Vialard, F. X., Amari, S. I., Trouvé, A., &
    Peyré, G. (2019, April). Interpolating between optimal transport and mmd
    using sinkhorn divergences. In The 22nd International Conference on
    Artificial Intelligence and Statistics (pp. 2681-2690). PMLR.

    [2] Feydy, J. Geometric Data Analysis, beyond Convolutions. Ph.D. Thesis,
    Université Paris-Saclay, Gif-sur-Yvette, France, 2020.

    Parameters
    ----------
    x       :  tf.Tensor of shape (num_draws_x, num_features)
        Comprises `num_draws_x` Random draws from the "source" distribution `P`.
    y       :  tf.Tensor of shape (num_draws_y, num_features)
        Comprises `num_draws_y` Random draws from the "source" distribution `Q`.

    Returns
    -------
    kernel  : tf.Tensor of shape (num_draws_x, num_draws_y)
        The kernel matrix between pairs from `x` and `y`.
    """

    norm = lambda v: tf.reduce_sum(tf.square(v), 1)
    d_xx = tf.expand_dims(norm(x),-1)
    d_xy = tf.matmul(x,tf.transpose(y))
    d_yy = tf.expand_dims(norm(y), 0)
    squared_dist = d_xx - 2 * d_xy + d_yy
    dist = tf.math.sqrt(tf.clip_by_value(squared_dist,
                                         clip_value_min=1e-8,
                                         clip_value_max=int(1e10)))
    kernel = - dist

    return kernel

def inverse_multiquadratic_kernel_matrix(x, y, sigmas=None):
    """Computes an inverse multiquadratic RBF between the samples of x and y.

    We create a sum of multiple IM-RBF kernels each having a width :math:`\sigma_i`.

    Parameters
    ----------
    x       :  tf.Tensor of shape (num_draws_x, num_features)
        Comprises `num_draws_x` Random draws from the "source" distribution `P`.
    y       :  tf.Tensor of shape (num_draws_y, num_features)
        Comprises `num_draws_y` Random draws from the "source" distribution `Q`.
    sigmas  : list(float), optional, default: None
        List which denotes the widths of each of the gaussians in the kernel.
        If `sigmas is None`, a default range will be used, contained in `bayesflow.default_settings.MMD_BANDWIDTH_LIST`

    Returns
    -------
    kernel  : tf.Tensor of shape (num_draws_x, num_draws_y)
        The kernel matrix between pairs from `x` and `y`.
    """

    if sigmas is None:
        sigmas = MMD_BANDWIDTH_LIST
    dist = tf.expand_dims(tf.reduce_sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1), axis=-1)
    sigmas = tf.expand_dims(sigmas, 0)
    return tf.reduce_sum(sigmas / (dist + sigmas), axis=-1)


def mmd_kernel(x, y, kernel):
    """Computes the estimator of the Maximum Mean Discrepancy (MMD) between two samples: x and y.

    Maximum Mean Discrepancy (MMD) is a distance-measure between random draws from
    the distributions `x ~ P` and `y ~ Q`.

    Parameters
    ----------
    x      : tf.Tensor of shape (N, num_features)
        An array of `N` random draws from the "source" distribution `x ~ P`.
    y      : tf.Tensor of shape (M, num_features)
        An array of `M` random draws from the "target" distribution `y ~ Q`.
    kernel : callable
        A function which computes the distance between pairs of samples.

    Returns
    -------
    loss   : tf.Tensor of shape (,)
        The statistically biased squared maximum mean discrepancy (MMD) value.
    """

    loss = tf.reduce_mean(kernel(x, x))
    loss += tf.reduce_mean(kernel(y, y))
    loss -= 2 * tf.reduce_mean(kernel(x, y))
    return loss


def mmd_kernel_unbiased(x, y, kernel):
    """Computes the unbiased estimator of the Maximum Mean Discrepancy (MMD) between two samples: x and y.
    Maximum Mean Discrepancy (MMD) is a distance-measure between the samples of the distributions `x ~ P` and `y ~ Q`.

    Parameters
    ----------
    x      : tf.Tensor of shape (N, num_features)
        An array of `N` random draws from the "source" distribution `x ~ P`.
    y      : tf.Tensor of shape (M, num_features)
        An array of `M` random draws from the "target" distribution `y ~ Q`.
    kernel : callable
        A function which computes the distance between pairs of random draws from `x` and `y`.

    Returns
    -------
    loss   : tf.Tensor of shape (,)
        The statistically unbiased squared maximum mean discrepancy (MMD) value.
    """

    m, n = x.shape[0], y.shape[0]
    loss = (1.0 / (m * (m + 1))) * tf.reduce_sum(kernel(x, x))
    loss += (1.0 / (n * (n + 1))) * tf.reduce_sum(kernel(y, y))
    loss -= (2.0 / (m * n)) * tf.reduce_sum(kernel(x, y))
    return loss


def expected_calibration_error(m_true, m_pred, num_bins=10):
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
        y_true = (m_true.argmax(axis=1) == k).astype(np.float32)
        y_prob = m_pred[:, k]
        prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=num_bins)

        # Compute ECE by weighting bin errors by bin size
        bins = np.linspace(0.0, 1.0, num_bins + 1)
        binids = np.searchsorted(bins[1:-1], y_prob)
        bin_total = np.bincount(binids, minlength=len(bins))
        nonzero = bin_total != 0
        cal_err = np.sum(np.abs(prob_true - prob_pred) * (bin_total[nonzero] / len(y_true)))

        cal_errs.append(cal_err)
        probs_true.append(prob_true)
        probs_pred.append(prob_pred)
    return cal_errs, probs_true, probs_pred


def maximum_mean_discrepancy(source_samples, target_samples, kernel="gaussian", mmd_weight=1.0, minimum=0.0):
    """Computes the MMD given a particular choice of kernel.

    For details, consult Gretton et al. (2012):
    https://www.jmlr.org/papers/volume13/gretton12a/gretton12a.pdf

    Parameters
    ----------
    source_samples : tf.Tensor of shape (N, num_features)
        An array of `N` random draws from the "source" distribution.
    target_samples : tf.Tensor of shape  (M, num_features)
        An array of `M` random draws from the "target" distribution.
    kernel         : str in ('gaussian', 'inverse_multiquadratic'), optional, default: 'gaussian'
        The kernel to use for computing the distance between pairs of random draws.
    mmd_weight     : float, optional, default: 1.0
        The weight of the MMD value.
    minimum        : float, optional, default: 0.0
        The lower bound of the MMD value.

    Returns
    -------
    loss_value : tf.Tensor
        A scalar Maximum Mean Discrepancy, shape (,)
    """

    # Determine kernel, fall back to Gaussian if unknown string passed
    if kernel == "gaussian":
        kernel_fun = gaussian_kernel_matrix
    elif kernel == "inverse_multiquadratic":
        kernel_fun = inverse_multiquadratic_kernel_matrix
    elif kernel == "energy":
        kernel_fun = energy_kernel_matrix
    else:
        kernel_fun = gaussian_kernel_matrix

    # Compute and return MMD value
    loss_value = mmd_kernel(source_samples, target_samples, kernel=kernel_fun)
    loss_value = mmd_weight * tf.maximum(minimum, loss_value)
    return loss_value


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
    F_m = np.sum((z[:, np.newaxis] >= u[:, np.newaxis, :]), axis=-1) / u.shape[1]
    bin1 = stats.binom(N, z).cdf(N * F_m)
    bin2 = stats.binom(N, z).cdf(N * F_m - 1)
    gamma = 2 * np.min(np.min(np.stack([bin1, 1 - bin2], axis=-1), axis=-1), axis=-1)
    return gamma


def simultaneous_ecdf_bands(
    num_samples, num_points=None, num_simulations=1000, confidence=0.95, eps=1e-5, max_num_points=1000
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
    z = np.linspace(0 + eps, 1 - eps, K)

    # Simulate M samples of size N
    u = np.random.uniform(size=(M, N))

    # Get alpha
    alpha = 1 - confidence

    # Compute minimal coverage probabilities
    gammas = get_coverage_probs(z, u)

    # Use insights from paper to compute lower and upper confidence interval
    gamma = np.percentile(gammas, 100 * alpha)
    L = stats.binom(N, z).ppf(gamma / 2) / N
    U = stats.binom(N, z).ppf(1 - gamma / 2) / N
    return alpha, z, L, U


def mean_squared_error(x_true, x_pred):
    """Computes the mean squared error between a single true value and M estimates thereof.

    Parameters
    ----------

    x_true      : float or np.ndarray
        true values, shape ()
    x_pred      : np.ndarray
        predicted values, shape (M, )

    Returns
    -------
    out : float
        The MSE between ``x_true`` and ``x_pred``
    """

    x_true = np.array(x_true)
    x_pred = np.array(x_pred)
    try:
        return np.mean((x_true[np.newaxis, :] - x_pred) ** 2)
    except IndexError:
        return np.mean((x_true - x_pred) ** 2)


def root_mean_squared_error(x_true, x_pred):
    """Computes the root mean squared error (RMSE) between a single true value and M estimates thereof.

    Parameters
    ----------

    x_true      : float or np.ndarray
        true values, shape ()
    x_pred      : np.ndarray
        predicted values, shape (M, )

    Returns
    -------
    out : float
        The RMSE between ``x_true`` and ``x_pred``
    """

    mse = mean_squared_error(x_true=x_true, x_pred=x_pred)
    return np.sqrt(mse)


def aggregated_error(x_true, x_pred, inner_error_fun=root_mean_squared_error, outer_aggregation_fun=np.mean):
    """Computes the aggregated error between a vector of N true values and M estimates of each true value.

    x_true      : np.ndarray
    true values, shape (N)

    x_pred      : np.ndarray
    predicted values, shape (M, N)

    inner_error_fun: callable, default: root_mean_squared_error
    computes the error between one true value and M estimates thereof

    outer_aggregation_fun: callable, default: np.mean
    aggregates N errors to a single aggregated error value
    """

    x_true, x_pred = np.array(x_true), np.array(x_pred)

    N = x_pred.shape[0]
    if not N == x_true.shape[0]:
        raise ShapeError

    errors = np.array([inner_error_fun(x_true=x_true[i], x_pred=x_pred[i]) for i in range(N)])

    if not N == errors.shape[0]:
        raise ShapeError

    return outer_aggregation_fun(errors)


def aggregated_rmse(x_true, x_pred):
    """
    Computes the aggregated RMSE for a matrix of predictions.

    Parameters
    ----------
    x_true      : np.ndarray
    true values, shape (N)

    x_pred      : np.ndarray
    predicted values, shape (M, N)

    Returns
    -------
    aggregated RMSE
    """

    return aggregated_error(
        x_true=x_true, x_pred=x_pred, inner_error_fun=root_mean_squared_error, outer_aggregation_fun=np.mean
    )


def c2st(
    source_samples,
    target_samples,
    n_folds=5,
    scoring="accuracy",
    normalize=True,
    seed=123,
    hidden_units_per_dim=16,
    aggregate_output=True,
):
    """C2ST metric [1] using an sklearn neural network classifier (i.e., MLP).
    Code adapted from https://github.com/sbi-benchmark/sbibm/blob/main/sbibm/metrics/c2st.py

    [1] Lopez-Paz, D., & Oquab, M. (2016). Revisiting classifier two-sample tests. arXiv:1610.06545.

    Parameters
    ----------
    source_samples       : np.ndarray or tf.Tensor
        Source samples (e.g., approximate posterior samples)
    target_samples       : np.ndarray or tf.Tensor
        Target samples (e.g., samples from a reference posterior)
    n_folds              : int, optional, default: 5
        Number of folds in k-fold cross-validation for the classifier evaluation
    scoring              : str, optional, default: "accuracy"
        Evaluation score of the sklearn MLP classifier
    normalize            : bool, optional, default: True
        Whether the data shall be z-standardized relative to source_samples
    seed                 : int, optional, default: 123
        RNG seed for the MLP and k-fold CV
    hidden_units_per_dim : int, optional, default: 16
        Number of hidden units in the MLP, relative to the input dimensions.
        Example: source samples are 5D, hidden_units_per_dim=16 -> 80 hidden units per layer
    aggregate_output     : bool, optional, default: True
        Whether to return a single value aggregated over all cross-validation runs
        or all values from all runs. If left at default, the empirical mean will be returned

    Returns
    -------
    c2st_score  :  float
        The resulting C2ST score

    """

    x = np.array(source_samples)
    y = np.array(target_samples)

    num_dims = x.shape[1]
    if not num_dims == y.shape[1]:
        raise ShapeError(
            f"source_samples and target_samples can have different number of observations (1st dim)"
            f"but must have the same dimensionality (2nd dim)"
            f"found: source_samples {source_samples.shape[1]}, target_samples {target_samples.shape[1]}"
        )

    if normalize:
        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x = (x - x_mean) / x_std
        y = (y - x_mean) / x_std

    clf = MLPClassifier(
        activation="relu",
        hidden_layer_sizes=(hidden_units_per_dim * num_dims, hidden_units_per_dim * num_dims),
        max_iter=10000,
        solver="adam",
        random_state=seed,
    )

    data = np.concatenate((x, y))
    target = np.concatenate(
        (
            np.zeros((x.shape[0],)),
            np.ones((y.shape[0],)),
        )
    )

    shuffle = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, data, target, cv=shuffle, scoring=scoring)

    if aggregate_output:
        c2st_score = np.asarray(np.mean(scores)).astype(np.float32)
    return c2st_score
