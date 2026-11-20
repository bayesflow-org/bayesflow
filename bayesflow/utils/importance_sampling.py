from typing import Any

import numpy as np
from scipy.special import logsumexp

from bayesflow.types import LogProb

from . import logging
from .dict_utils import gather_maybe_nested


def psis_resample(
    samples,
    prior_log_prob: LogProb,
    likelihood_log_prob: LogProb,
    posterior_log_prob: LogProb,
    *,
    r_eff: float = 1.0,
    rng: int | np.random.Generator | None = None,
    return_diagnostics: bool = False,
):
    """Correct approximate posterior samples with Pareto-smoothed importance sampling.

    This is useful when an amortized posterior is fast to sample from but imperfect for
    a particular observation, while the prior and likelihood can be evaluated. The
    function forms the importance ratios

    ``prior * likelihood / approximate_posterior``,

    Pareto-smooths their upper tail, and resamples the joint draws. Large Pareto
    :math:`k` values trigger a log warning because they indicate that the approximate
    posterior has insufficient overlap with the target posterior.

    Log-probability arguments can be arrays or callables receiving ``samples``. Their
    output shape must be ``(*batch_shape, num_samples)``. Sample arrays must begin with
    the same dimensions, so native BayesFlow output with shape
    ``(num_conditions, num_samples, ...)`` works directly. If ``samples`` is a mapping,
    all values are resampled with shared indices to preserve joint draws.

    The posterior log density is required: prior and likelihood values alone do not
    define importance ratios for draws from an approximate posterior. All three log
    densities must be evaluated with respect to the same parameterization and measure;
    callers are responsible for any model-specific transformations and Jacobians.

    Parameters
    ----------
    samples : np.ndarray or mapping of np.ndarray
        Samples to correct. Leading dimensions must match the log-probability shape.
    prior_log_prob : np.ndarray or callable
        Prior log density at each draw.
    likelihood_log_prob : np.ndarray or callable
        Likelihood log density at each draw for the observation being conditioned on.
    posterior_log_prob : np.ndarray or callable
        Approximate-posterior log density at each draw.
    r_eff : float, optional
        Relative sampling efficiency used to select the Pareto tail size. Independent
        BayesFlow samples use the default of ``1.0``.
    rng : int or np.random.Generator, optional
        Local random generator or seed used for systematic resampling.
    return_diagnostics : bool, optional
        If ``True``, also return a dictionary containing ``"pareto_k"`` with shape
        ``batch_shape``.

    Returns
    -------
    corrected_samples
        Resampled values with the same structure and shapes as ``samples``.
    diagnostics : dict, optional
        Returned only when ``return_diagnostics=True``.

    References
    ----------
    Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024).
    Pareto smoothed importance sampling. Journal of Machine Learning Research, 25(72), 1-58.
    """
    if not 0 < r_eff <= 1:
        raise ValueError(f"r_eff must be in (0, 1], got {r_eff}.")

    log_prior = _evaluate_log_prob(prior_log_prob, samples, "prior_log_prob")
    log_likelihood = _evaluate_log_prob(likelihood_log_prob, samples, "likelihood_log_prob")
    log_posterior = _evaluate_log_prob(posterior_log_prob, samples, "posterior_log_prob")
    _check_log_prob_shapes(log_prior, log_likelihood, log_posterior)

    log_weights, pareto_k = _pareto_smooth(log_prior + log_likelihood - log_posterior, r_eff)

    threshold = min(1 - 1 / np.log10(log_weights.shape[-1]), 0.7)
    unreliable = np.isnan(pareto_k) | (pareto_k > threshold)

    if np.any(unreliable):
        logging.warning(
            f"Pareto k exceeds {threshold:.2f} or could not be estimated for "
            f"{np.count_nonzero(unreliable)} batch element(s); the PSIS correction may be unreliable.",
        )

    corrected = _resample(samples, log_weights, rng)
    if return_diagnostics:
        return corrected, {"pareto_k": pareto_k}
    return corrected


def importance_resample(
    samples,
    old_prior_log_prob: LogProb,
    new_prior_log_prob: LogProb,
    *,
    rng: int | np.random.Generator | None = None,
):
    """Reuse posterior samples after changing the prior.

    If ``samples`` approximate a posterior under an old prior, reweighting by
    ``new_prior / old_prior`` produces a sampling-importance-resampling approximation
    to the posterior under the new prior without retraining the estimator. The
    likelihood must remain unchanged, and the old prior must cover the support of the
    new prior wherever the new posterior has appreciable mass.

    Log-probability arguments can be arrays or callables receiving ``samples``. Their
    output shape must be ``(*batch_shape, num_samples)``. Sample arrays must begin with
    the same dimensions, so native BayesFlow output with shape
    ``(num_conditions, num_samples, ...)`` works directly. Model-specific parameter
    transformations and density Jacobians belong in the log-probability callables.

    Parameters
    ----------
    samples : np.ndarray or mapping of np.ndarray
        Posterior samples under the old prior.
    old_prior_log_prob : np.ndarray or callable
        Old-prior log density at each draw.
    new_prior_log_prob : np.ndarray or callable
        New-prior log density at each draw.
    rng : int or np.random.Generator, optional
        Local random generator or seed used for systematic resampling.

    Returns
    -------
    resampled
        Values with the same structure and shapes as ``samples``. Mapping values are
        gathered with shared indices to preserve joint draws.
    """
    old_log_prob = _evaluate_log_prob(old_prior_log_prob, samples, "old_prior_log_prob")
    new_log_prob = _evaluate_log_prob(new_prior_log_prob, samples, "new_prior_log_prob")
    _check_log_prob_shapes(old_log_prob, new_log_prob)

    log_weights = _normalize_log_weights(new_log_prob - old_log_prob)
    return _resample(samples, log_weights, rng)


def _evaluate_log_prob(log_prob: LogProb, samples, name: str) -> np.ndarray:
    """Evaluate a callable log density or use precomputed values."""
    values = log_prob(samples) if callable(log_prob) else log_prob
    if values.ndim == 0:
        raise ValueError(f"{name} must have a sample dimension.")
    if np.any(np.isnan(values)):
        raise ValueError(f"{name} contains NaN values.")
    if np.any(np.isposinf(values)):
        raise ValueError(f"{name} contains positive infinity.")
    return values


def _check_log_prob_shapes(*log_probs: np.ndarray) -> None:
    """Check that all log densities describe the same draws."""
    expected = log_probs[0].shape
    for log_prob in log_probs[1:]:
        if log_prob.shape != expected:
            raise ValueError(
                f"All log-probability arrays must have the same shape, got {expected} and {log_prob.shape}."
            )


def _normalize_log_weights(log_weights: np.ndarray) -> np.ndarray:
    """Normalize log weights along the sample axis."""
    if np.any(np.isnan(log_weights)) or np.any(np.isposinf(log_weights)):
        raise ValueError("Log importance ratios must not contain NaN or positive infinity.")

    normalizer = logsumexp(log_weights, axis=-1, keepdims=True)
    if np.any(~np.isfinite(normalizer)):
        raise ValueError("Each batch element must contain at least one finite importance ratio.")
    return log_weights - normalizer


def _pareto_smooth(log_ratios: np.ndarray, r_eff: float) -> tuple[np.ndarray, np.ndarray]:
    """Pareto-smooth each batch element independently."""
    if log_ratios.shape[-1] < 10:
        raise ValueError("PSIS requires at least 10 samples per batch element.")
    if np.any(np.isnan(log_ratios)) or np.any(np.isposinf(log_ratios)):
        raise ValueError("Log importance ratios must not contain NaN or positive infinity.")

    flat = log_ratios.reshape(-1, log_ratios.shape[-1])
    smoothed = np.empty_like(flat, dtype=float)
    pareto_k = np.empty(flat.shape[0])
    for index, row in enumerate(flat):
        smoothed[index], pareto_k[index] = _pareto_smooth_row(row, r_eff)

    batch_shape = log_ratios.shape[:-1]
    return smoothed.reshape(log_ratios.shape), pareto_k.reshape(batch_shape)


def _pareto_smooth_row(log_ratios: np.ndarray, r_eff: float) -> tuple[np.ndarray, float]:
    """Pareto-smooth one vector of log importance ratios."""
    finite = np.isfinite(log_ratios)
    if not np.any(finite):
        raise ValueError("Each batch element must contain at least one finite importance ratio.")

    centered = log_ratios - np.max(log_ratios)
    finite_values = centered[finite]
    if np.ptp(finite_values) <= np.finfo(float).eps:
        return _normalize_log_weights(centered), -np.inf

    num_samples = centered.size
    if num_samples * r_eff > 225:
        tail_size = int(np.floor(3 * np.sqrt(num_samples / r_eff)))
    else:
        tail_size = int(np.floor(num_samples / 5))
    tail_size = max(tail_size, 5)

    order = np.argsort(centered)
    tail_indices = order[-tail_size:]
    cutoff = centered[order[-tail_size - 1]]
    tail = centered[tail_indices]

    if not np.isfinite(cutoff) or np.ptp(tail) <= np.finfo(float).eps:
        return _normalize_log_weights(centered), np.nan

    cutoff_weight = np.exp(cutoff)
    excess = np.exp(tail) - cutoff_weight
    pareto_k, scale = _fit_generalized_pareto(excess)

    if np.isfinite(pareto_k) and np.isfinite(scale) and scale > 0:
        probabilities = (np.arange(tail_size) + 0.5) / tail_size
        smoothed_tail = cutoff_weight + _generalized_pareto_quantile(probabilities, pareto_k, scale)
        smoothed_tail = np.minimum(np.log(smoothed_tail), tail[-1])
        centered = centered.copy()
        centered[tail_indices] = smoothed_tail

    return _normalize_log_weights(centered), pareto_k


def _fit_generalized_pareto(excess: np.ndarray) -> tuple[float, float]:
    """Fit a generalized Pareto tail using the empirical-Bayes PSIS estimator."""
    num_tail = excess.size
    first_quartile = excess[int(num_tail / 4 + 0.5) - 1]
    if first_quartile <= excess[0] or excess[-1] <= 0:
        return np.nan, np.nan

    num_candidates = 30 + int(np.sqrt(num_tail))
    candidates = 1 - np.sqrt(num_candidates / (np.arange(1, num_candidates + 1) - 0.5))
    candidates /= 3 * first_quartile
    candidates += 1 / excess[-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        shapes = np.mean(np.log1p(-candidates[:, None] * excess), axis=1)
        log_likelihood = num_tail * (np.log(-candidates / shapes) - shapes - 1)

    valid = np.isfinite(log_likelihood)
    if not np.any(valid):
        return np.nan, np.nan
    candidates = candidates[valid]
    log_likelihood = log_likelihood[valid]
    candidate_weights = np.exp(log_likelihood - logsumexp(log_likelihood))
    keep = candidate_weights >= 10 * np.finfo(float).eps
    candidates = candidates[keep]
    candidate_weights = candidate_weights[keep]
    candidate_weights /= candidate_weights.sum()

    fitted_candidate = np.sum(candidates * candidate_weights)
    with np.errstate(divide="ignore", invalid="ignore"):
        shape = np.mean(np.log1p(-fitted_candidate * excess))
    scale = -shape / fitted_candidate
    shape = (num_tail * shape + 5) / (num_tail + 10)
    return shape, scale


def _generalized_pareto_quantile(probabilities: np.ndarray, shape: float, scale: float) -> np.ndarray:
    """Evaluate generalized Pareto quantiles."""
    if abs(shape) < np.finfo(float).eps:
        return -scale * np.log1p(-probabilities)
    return scale * np.expm1(-shape * np.log1p(-probabilities)) / shape


def _resample(samples, normalized_log_weights: np.ndarray, rng) -> Any:
    """Systematically resample every leaf with shared indices."""
    generator = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    num_samples = normalized_log_weights.shape[-1]
    weights = np.exp(normalized_log_weights).reshape(-1, num_samples)
    offsets = generator.random(weights.shape[0])[:, None]
    positions = (offsets + np.arange(num_samples)) / num_samples
    cumulative = np.cumsum(weights, axis=1)
    cumulative[:, -1] = 1.0

    flat_indices = np.empty_like(positions, dtype=int)
    for index in range(weights.shape[0]):
        flat_indices[index] = np.searchsorted(cumulative[index], positions[index], side="right")
    indices = flat_indices.reshape(normalized_log_weights.shape)
    return gather_maybe_nested(samples, indices)
