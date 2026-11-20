from typing import Any, Literal

import numpy as np
from scipy.special import logsumexp

from bayesflow.types import LogProb

from . import logging
from .dict_utils import gather_maybe_nested


def importance_resample(
    samples,
    log_ratio: LogProb,
    *,
    method: Literal["psis", "is"] = "psis",
    r_eff: float = 1.0,
    rng: int | np.random.Generator | None = None,
    return_diagnostics: bool = False,
):
    """Correct posterior samples by importance resampling.

    ``log_ratio`` is the log density ratio between the target and current posterior,
    up to an additive normalization constant. It can be a precomputed array or a
    callable receiving ``samples``. For posterior approximation correction, compute

    .. code-block:: python

        log_ratio = lambda samples: (
            prior_log_prob(samples) + likelihood_log_prob(samples) - posterior_log_prob(samples)
        )

    For a prior shift, the unchanged likelihood cancels, so compute

    .. code-block:: python

        log_ratio = lambda samples: new_prior_log_prob(samples) - old_prior_log_prob(samples)

    The ratio must have shape ``(*batch_shape, num_samples)``. Sample arrays must
    begin with the same dimensions, so native BayesFlow output with shape
    ``(num_conditions, num_samples, ...)`` works directly. Mapping values are
    resampled with shared indices to preserve joint draws. Callers are responsible
    for using a common parameterization and including model-specific Jacobians.

    Parameters
    ----------
    samples : np.ndarray or mapping of np.ndarray
        Posterior samples to correct.
    log_ratio : np.ndarray or callable
        Target-to-current posterior log density ratio at each draw.
    method : {"psis", "is"}, optional
        ``"psis"`` Pareto-smooths the ratios before resampling. ``"is"`` uses
        the normalized ratios directly. Default is ``"psis"``.
    r_eff : float, optional
        Relative sampling efficiency used by ``method="psis"`` to select the
        Pareto tail size. Independent BayesFlow samples use the default ``1.0``.
    rng : int or np.random.Generator, optional
        Local random generator or seed used for systematic resampling.
    return_diagnostics : bool, optional
        If ``True``, also return diagnostics. For ``method="psis"``, these contain
        ``"pareto_k"`` with shape ``batch_shape``; for ``method="is"``, the
        diagnostics dictionary is empty.

    Returns
    -------
    resampled
        Values with the same structure and shapes as ``samples``. Mapping values are
        gathered with shared indices to preserve joint draws.
    diagnostics : dict, optional
        Returned only when ``return_diagnostics=True``.

    References
    ----------
    Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024).
    Pareto smoothed importance sampling. Journal of Machine Learning Research, 25(72), 1-58.
    """
    if method not in {"psis", "is"}:
        raise ValueError(f"method must be 'psis' or 'is', got {method!r}.")

    log_ratio = _evaluate_log_ratio(log_ratio, samples)
    diagnostics = {}

    if method == "psis":
        if not 0 < r_eff <= 1:
            raise ValueError(f"r_eff must be in (0, 1], got {r_eff}.")

        log_weights, pareto_k = _pareto_smooth(log_ratio, r_eff)
        diagnostics["pareto_k"] = pareto_k
        threshold = min(1 - 1 / np.log10(log_weights.shape[-1]), 0.7)
        unreliable = np.isnan(pareto_k) | (pareto_k > threshold)

        if np.any(unreliable):
            logging.warning(
                f"Pareto k exceeds {threshold:.2f} or could not be estimated for "
                f"{np.count_nonzero(unreliable)} batch element(s); the PSIS correction may be unreliable.",
            )
    else:
        log_weights = _normalize_log_weights(log_ratio)

    corrected = _resample(samples, log_weights, rng)
    if return_diagnostics:
        return corrected, diagnostics
    return corrected


def _evaluate_log_ratio(log_ratio: LogProb, samples) -> np.ndarray:
    """Evaluate a callable log ratio or use precomputed values."""
    values = log_ratio(samples) if callable(log_ratio) else log_ratio
    if values.ndim == 0:
        raise ValueError("log_ratio must have a sample dimension.")
    if np.any(np.isnan(values)):
        raise ValueError("log_ratio contains NaN values.")
    if np.any(np.isposinf(values)):
        raise ValueError("log_ratio contains positive infinity.")
    return values


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

    centered = log_ratios.astype(float, copy=False) - np.max(log_ratios)
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
