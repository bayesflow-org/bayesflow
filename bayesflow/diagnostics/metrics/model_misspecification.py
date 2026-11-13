"""
This module provides functions for computing distances between observation samples and reference samples with distance
distributions within the reference samples for hypothesis testing.
"""

import os
import warnings
from collections.abc import Mapping, Callable
from typing import NamedTuple

import numpy as np
import scipy.linalg

import keras

from bayesflow.estimators import BasicEstimator
from bayesflow.metrics.functional import maximum_mean_discrepancy
from bayesflow.types import Tensor


def bootstrap_comparison(
    observed_samples: np.ndarray,
    reference_samples: np.ndarray,
    comparison_fn: Callable[[Tensor, Tensor], Tensor],
    num_null_samples: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the distance between observed and reference samples and generates a distribution of null sample
    distances by bootstrapping for hypothesis testing.

    Parameters
    ----------
    observed_samples : np.ndarray
        Observed samples, shape (num_observed, ...).
    reference_samples : np.ndarray
        Reference samples, shape (num_reference, ...).
    comparison_fn : Callable[[Tensor, Tensor], Tensor]
        Function to compute the distance metric.
    num_null_samples : int
        Number of null samples to generate for hypothesis testing. Default is 100.

    Returns
    -------
    distance_observed : np.ndarray
        A scalar NumPy array containing the distance between observed and reference samples.
    distance_null : np.ndarray
        A distribution of distance values under the null hypothesis.

    Raises
    ------
    ValueError
        - If the number of number of observed samples exceeds the number of reference samples
        - If the shapes of observed and reference samples do not match on dimensions besides the first one.
    """
    observed_samples = np.asarray(observed_samples)
    reference_samples = np.asarray(reference_samples)

    num_observed = observed_samples.shape[0]
    num_reference = reference_samples.shape[0]

    if observed_samples.shape[1:] != reference_samples.shape[1:]:
        raise ValueError(
            f"Expected observed and reference samples to have the same shape, "
            f"but got {observed_samples.shape[1:]} != {reference_samples.shape[1:]}."
        )

    dtype = keras.config.floatx()
    observed_samples = keras.ops.convert_to_tensor(observed_samples, dtype=dtype)
    reference_samples = keras.ops.convert_to_tensor(reference_samples, dtype=dtype)

    distance_observed = comparison_fn(observed_samples, reference_samples)
    distance_observed = keras.ops.convert_to_numpy(distance_observed)

    if distance_observed.ndim != 0:
        raise ValueError(f"Expected comparison_fn to return a scalar, but got shape {distance_observed.shape}.")

    distance_null_samples = np.empty(num_null_samples, dtype=distance_observed.dtype)
    for i in range(num_null_samples):
        bootstrap_indices = np.random.randint(num_reference, size=num_observed)
        bootstrap_samples = keras.ops.take(reference_samples, bootstrap_indices, axis=0)
        distance_null = comparison_fn(bootstrap_samples, reference_samples)
        distance_null_samples[i] = keras.ops.convert_to_numpy(distance_null)

    return distance_observed, distance_null_samples


def summary_space_comparison(
    observed_data: Mapping[str, np.ndarray],
    reference_data: Mapping[str, np.ndarray],
    estimator: BasicEstimator,
    num_null_samples: int = 100,
    comparison_fn: Callable = maximum_mean_discrepancy,
    **kwargs,
) -> tuple[np.ndarray, np.ndarray]:
    """Computes the distance between observed and reference data in the summary space and
    generates a distribution of distance values under the null hypothesis to assess model misspecification.

    By default, the Maximum Mean Discrepancy (MMD) is used as a distance function.

    [1] M. Schmitt, P.-C. Bürkner, U. Köthe, and S. T. Radev, "Detecting model misspecification in amortized Bayesian
    inference with neural networks," arXiv e-prints, Dec. 2021, Art. no. arXiv:2112.08866.
    URL: https://arxiv.org/abs/2112.08866

    Parameters
    ----------
    observed_data : dict[str, np.ndarray]
        Dictionary of observed data as NumPy arrays, which will be preprocessed by the estimator's adapter and passed
        through its summary network.
    reference_data : dict[str, np.ndarray]
        Dictionary of reference data as NumPy arrays, which will be preprocessed by the estimator's adapter and passed
        through its summary network.
    estimator : BasicEstimator
        An instance of :py:class:`~bayesflow.estimators.BasicEstimator` used to compute summary statistics
        from the data.
    num_null_samples : int, optional
        Number of null samples to generate for hypothesis testing. Default is 100.
    comparison_fn : Callable, optional
        Distance function to compare the data in the summary space.
    **kwargs : dict
        Additional keyword arguments for the adapter and sampling process.

    Returns
    -------
    distance_observed : np.ndarray
        A scalar NumPy array containing the MMD value between observed and reference summaries.
    distance_null : np.ndarray
        A distribution of MMD values under the null hypothesis.

    Raises
    ------
    ValueError
        If estimator is not an instance of BasicEstimator or does not have a summary network.
    """

    if not isinstance(estimator, BasicEstimator):
        raise ValueError("The estimator must be an instance of BasicEstimator.")

    if not hasattr(estimator, "summary_network") or estimator.summary_network is None:
        comparison_fn_name = (
            "bayesflow.metrics.functional.maximum_mean_discrepancy"
            if comparison_fn is maximum_mean_discrepancy
            else comparison_fn.__name__
        )
        raise ValueError(
            "The estimator must have a summary network. If you have manually crafted summary "
            "statistics, or want to compare raw data and not summary statistics, please use the "
            f"`bootstrap_comparison` function with `comparison_fn={comparison_fn_name}` on the respective arrays."
        )

    observed_summaries = estimator.summarize(observed_data)
    reference_summaries = estimator.summarize(reference_data)

    distance_observed, distance_null = bootstrap_comparison(
        observed_samples=observed_summaries,
        reference_samples=reference_summaries,
        comparison_fn=comparison_fn,
        num_null_samples=num_null_samples,
    )

    return distance_observed, distance_null


def mahalanobis_distance(
    samples: np.ndarray,
    reference_samples: np.ndarray,
    *,
    regularization: float = 1e-6,
    squared: bool = False,
) -> np.ndarray:
    """Computes the Mahalanobis distance of each sample to the distribution of the reference samples.

    The mean and covariance are estimated from ``reference_samples``. Trailing dimensions are flattened, so each
    sample is treated as a vector of size ``prod(shape[1:])``. Computations are carried out in float64.

    Parameters
    ----------
    samples : np.ndarray
        Samples to score, shape (num_samples, ...).
    reference_samples : np.ndarray
        Samples used to estimate the mean and covariance, shape (num_reference, ...).
    regularization : float, optional
        Ridge term added to the diagonal of the covariance, relative to each dimension's variance (equivalently,
        added to the diagonal of the correlation matrix), so that the distance stays invariant to rescaling of
        individual dimensions. Keeps the covariance invertible when some dimensions are (nearly) collinear.
        Constant dimensions receive a ridge relative to the average variance. Default is 1e-6.
    squared : bool, optional
        Whether to return squared distances, e.g., to compare them with a chi-square distribution.
        Default is False.

    Returns
    -------
    distances : np.ndarray
        Mahalanobis distances, or squared distances if ``squared`` is True, shape (num_samples,).

    Raises
    ------
    ValueError
        If the shapes of samples and reference samples do not match on dimensions besides the first one,
        if there are fewer than two reference samples, if any input is not finite, if ``regularization``
        is negative, or if the covariance is singular and ``regularization`` is zero.
    """
    reference_samples = _as_matrix(reference_samples, "reference_samples")
    samples = _as_matrix(samples, "samples", reference_samples)

    return _distances(_fit_gaussian(reference_samples, regularization), samples, squared)


class _GaussianFit(NamedTuple):
    mean: np.ndarray
    cholesky: np.ndarray


def _as_matrix(x, name: str, reference: np.ndarray = None) -> np.ndarray:
    """Converts to a finite float64 array of shape (num_samples, prod(trailing dims)). If ``reference`` is given,
    the trailing dimensions must match those of ``reference``, which must already be converted."""
    if isinstance(x, Mapping):
        raise ValueError(
            f"Expected {name} to be an array of shape (num_samples, ...), but got a dict with keys {list(x)}. "
            "Pass summary statistics as arrays, or use `summary_space_mahalanobis` to summarize data dicts with "
            "an estimator."
        )

    x = np.asarray(keras.ops.convert_to_numpy(x), dtype=np.float64)

    if x.ndim == 0:
        raise ValueError(f"Expected {name} to have shape (num_samples, ...), but got a scalar.")

    if reference is not None and int(np.prod(x.shape[1:])) != reference.shape[1]:
        if x.ndim == 1 and x.size == reference.shape[1] > 1:
            hint = f" For a single sample, add a leading axis, e.g., `{name}[None]`, to get shape (1, {x.size})."
        else:
            hint = ""
        raise ValueError(
            f"Expected {name} to have shape (num_samples, {reference.shape[1]}) to match the reference samples, "
            f"but got {x.shape}.{hint}"
        )

    x = x.reshape(x.shape[0], int(np.prod(x.shape[1:])))

    if not np.all(np.isfinite(x)):
        raise ValueError(f"Expected {name} to be finite, but found nan or inf.")

    return x


def _fit_gaussian(reference_samples: np.ndarray, regularization: float) -> _GaussianFit:
    """Estimates the mean and the regularized covariance (as its Cholesky factor)."""
    if regularization < 0:
        raise ValueError(f"Expected regularization to be non-negative, but got {regularization}.")

    num_reference, dim = reference_samples.shape

    if num_reference < 2:
        raise ValueError(f"Expected at least two reference samples, but got {num_reference}.")

    if num_reference <= dim:
        warnings.warn(
            f"The number of reference samples ({num_reference}) does not exceed the dimension ({dim}), so the "
            "covariance is singular and distances are dominated by the regularization. Use more reference samples.",
            UserWarning,
            skip_file_prefixes=(os.path.dirname(__file__),),
        )

    mean = np.mean(reference_samples, axis=0)
    covariance = np.atleast_2d(np.cov(reference_samples, rowvar=False))

    if regularization > 0:
        # detect constant dimensions on the data, since np.cov can return round-off variances for them
        is_constant = np.ptp(reference_samples, axis=0) == 0
        variances = np.where(is_constant, 0.0, np.diag(covariance))
        # relative to each dimension's variance to keep distances invariant to rescaling of dimensions;
        # constant dimensions have no scale of their own, so use the average variance (or 1 if all are constant)
        fallback = np.mean(variances[~is_constant]) if not np.all(is_constant) else 1.0
        covariance = covariance + regularization * np.diag(np.where(is_constant, fallback, variances))

    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError(
            "The covariance of the reference samples is singular. Use regularization > 0 or more reference samples."
        ) from error

    return _GaussianFit(mean=mean, cholesky=cholesky)


def _distances(fit: _GaussianFit, samples: np.ndarray, squared: bool) -> np.ndarray:
    whitened = scipy.linalg.solve_triangular(fit.cholesky, (samples - fit.mean).T, lower=True)
    squared_distances = np.sum(whitened**2, axis=0)
    return squared_distances if squared else np.sqrt(squared_distances)


def mahalanobis_test(
    observed_samples: np.ndarray,
    reference_samples: np.ndarray,
    calibration_samples: np.ndarray,
    *,
    alpha: float = 0.05,
    regularization: float = 1e-6,
    squared: bool = False,
) -> dict[str, np.ndarray]:
    """Flags out-of-distribution (OOD) samples by their Mahalanobis distance to the reference samples.

    The mean and covariance are estimated from ``reference_samples``, and each observed sample is tested
    individually at level ``alpha`` against the distances of ``calibration_samples``, i.e., held-out draws from the
    reference distribution that were not used to estimate mean and covariance. Thresholding at a quantile of the
    distances of simulated data follows [1]. Here, the p-value of a distance ``d`` is the conformal p-value
    ``(1 + #{calibration distances >= d}) / (num_calibration + 1)``, which is valid for any sample size and makes no
    distributional assumption. Summary spaces trained towards a Gaussian, e.g., with
    ``SummaryNetwork(base_distribution="normal")`` [2], make the Mahalanobis distance a more informative score,
    but are not required for a calibrated test.

    [1] Li, C., Vehtari, A., Bürkner, P.-C., Radev, S. T., Acerbi, L., & Schmitt, M. (2024). Amortized Bayesian
    workflow. arXiv:2409.04332.

    [2] Schmitt, M., Bürkner, P.-C., Köthe, U., & Radev, S. T. (2023). Detecting model misspecification in amortized
    Bayesian inference with neural networks. DAGM German Conference on Pattern Recognition, 541-557.

    Parameters
    ----------
    observed_samples : np.ndarray
        Samples to test, shape (num_observed, ...).
    reference_samples : np.ndarray
        Samples used to estimate the mean and covariance, shape (num_reference, ...).
    calibration_samples : np.ndarray
        Held-out samples from the reference distribution, shape (num_calibration, ...).
    alpha : float, optional
        Significance level of each test. Default is 0.05.
    regularization : float, optional
        Relative ridge term for the covariance, see :py:func:`mahalanobis_distance`. Default is 1e-6.
    squared : bool, optional
        Whether ``"distances"`` and ``"threshold"`` are squared distances. P-values and flags do not depend on it.
        Default is False.

    Returns
    -------
    result : dict[str, np.ndarray]
        Dictionary with the following entries:

        - ``"distances"``: Mahalanobis distances (squared if ``squared`` is True) of the observed samples,
          shape (num_observed,).
        - ``"p_values"``: p-values of the observed samples, shape (num_observed,).
        - ``"threshold"``: scalar critical value on the same scale as ``"distances"``; a sample is flagged if its
          distance exceeds it.
          Infinite if there are too few calibration samples for level ``alpha``, i.e., fewer than ``1 / alpha - 1``.
        - ``"is_ood"``: boolean array, shape (num_observed,), True where ``p_values <= alpha``.

    Raises
    ------
    ValueError
        If ``alpha`` is not in (0, 1), or if ``calibration_samples`` is empty or does not match the shape of the
        reference samples. Also raised for invalid inputs, see :py:func:`mahalanobis_distance`.
    """
    if not 0 < alpha < 1:
        raise ValueError(f"Expected alpha to be in (0, 1), but got {alpha}.")

    reference_samples = _as_matrix(reference_samples, "reference_samples")
    observed_samples = _as_matrix(observed_samples, "observed_samples", reference_samples)
    calibration_samples = _as_matrix(calibration_samples, "calibration_samples", reference_samples)

    if calibration_samples.shape[0] == 0:
        raise ValueError(
            "Expected non-empty calibration samples, i.e., held-out samples from the reference distribution."
        )

    # p-values and flags are invariant to the monotone square root; computing both observed and calibration
    # distances on the same scale keeps the threshold consistent with the flags
    fit = _fit_gaussian(reference_samples, regularization)
    distances = _distances(fit, observed_samples, squared)

    calibration_distances = np.sort(_distances(fit, calibration_samples, squared))
    num_calibration = calibration_distances.shape[0]
    num_greater_equal = num_calibration - np.searchsorted(calibration_distances, distances, side="left")
    p_values = (1 + num_greater_equal) / (num_calibration + 1)

    # the critical value is the smallest calibration distance such that any larger distance has p <= alpha;
    # computing it with the same arithmetic as the p-values keeps both consistent under round-off
    num_greater = num_calibration - np.searchsorted(calibration_distances, calibration_distances, side="right")
    rejects_above = (1 + num_greater) / (num_calibration + 1) <= alpha
    critical_value = calibration_distances[np.argmax(rejects_above)] if np.any(rejects_above) else np.inf

    return {
        "distances": distances,
        "p_values": p_values,
        "threshold": np.asarray(critical_value),
        "is_ood": p_values <= alpha,
    }


def summary_space_mahalanobis(
    observed_data: Mapping[str, np.ndarray],
    reference_data: Mapping[str, np.ndarray],
    estimator: BasicEstimator,
    *,
    calibration_data: Mapping[str, np.ndarray],
    alpha: float = 0.05,
    regularization: float = 1e-6,
    squared: bool = False,
) -> dict[str, np.ndarray]:
    """Scores observed data sets as out-of-distribution (OOD) by their Mahalanobis distance in summary space.

    Observed, reference, and calibration data are passed through the estimator's adapter and summary network, and
    the summaries are tested with :py:func:`mahalanobis_test`, which thresholds distances at a quantile of the
    distances of held-out simulated data [1]. Unlike :py:func:`summary_space_comparison`, which tests a whole set of
    observations at once, this scores each observed data set individually. A summary network trained with
    ``base_distribution="normal"`` [2] makes the distance more informative, but is not required.

    [1] Li, C., Vehtari, A., Bürkner, P.-C., Radev, S. T., Acerbi, L., & Schmitt, M. (2024). Amortized Bayesian
    workflow. arXiv:2409.04332.

    [2] Schmitt, M., Bürkner, P.-C., Köthe, U., & Radev, S. T. (2023). Detecting model misspecification in amortized
    Bayesian inference with neural networks. DAGM German Conference on Pattern Recognition, 541-557.

    Parameters
    ----------
    observed_data : dict[str, np.ndarray]
        Observed data sets to score.
    reference_data : dict[str, np.ndarray]
        Simulated data sets used to estimate the summary mean and covariance, e.g., (a subset of) the training data.
    estimator : BasicEstimator
        An instance of :py:class:`~bayesflow.estimators.BasicEstimator` with a summary network.
    calibration_data : dict[str, np.ndarray]
        Held-out simulated data sets, e.g., validation data, not part of ``reference_data``.
    alpha : float, optional
        Significance level of each test. Default is 0.05.
    regularization : float, optional
        Relative ridge term for the covariance, see :py:func:`mahalanobis_distance`. Default is 1e-6.
    squared : bool, optional
        Whether to report squared distances, see :py:func:`mahalanobis_test`. Default is False.

    Returns
    -------
    result : dict[str, np.ndarray]
        Dictionary with entries ``"distances"``, ``"p_values"``, ``"threshold"``, and ``"is_ood"``,
        see :py:func:`mahalanobis_test`.

    Raises
    ------
    ValueError
        If the estimator is not an instance of BasicEstimator or does not have a summary network, or if the
        arguments are invalid (see :py:func:`mahalanobis_test`).
    """
    if not isinstance(estimator, BasicEstimator):
        hint = (
            " Pass the calibration data by keyword, `calibration_data=...`, after the estimator."
            if isinstance(estimator, Mapping)
            else ""
        )
        raise ValueError(
            f"The estimator must be an instance of BasicEstimator, but got {type(estimator).__name__}.{hint}"
        )

    if getattr(estimator, "summary_network", None) is None:
        raise ValueError(
            "The estimator must have a summary network. If you have manually crafted summary statistics, or want "
            "to compare raw data, please use `mahalanobis_test` on the respective arrays."
        )

    return mahalanobis_test(
        estimator.summarize(observed_data),
        estimator.summarize(reference_data),
        estimator.summarize(calibration_data),
        alpha=alpha,
        regularization=regularization,
        squared=squared,
    )
