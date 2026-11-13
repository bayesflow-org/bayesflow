import warnings
from collections.abc import Callable, Mapping

import keras
import numpy as np
from scipy.special import logsumexp

from bayesflow.estimators import BasicEstimator


def pareto_smooth(log_ratios: np.ndarray, *, r_eff: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Pareto-smooths importance ratios and returns normalized log weights and the Pareto k diagnostic [1].

    The largest importance ratios are replaced by quantiles of a generalized Pareto distribution fitted to the
    upper tail, which stabilizes importance sampling estimates. The estimated shape parameter k of the fit
    diagnoses the reliability of the importance sampling estimate:

    - k < min(1 - 1 / log10(S), 0.7), with S the number of draws: estimates are reliable.
    - Otherwise, the proposal is too different from the target, and estimates are unreliable.

    Requires the optional dependency ``arviz-stats`` (>=1.1.0), installable with ``pip install arviz-stats``.

    [1] Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024). Pareto smoothed importance sampling.
    Journal of Machine Learning Research, 25(72), 1-58.

    Parameters
    ----------
    log_ratios : np.ndarray
        Log importance ratios, e.g., ``log p(theta, x) - log q(theta | x)``, shape (..., num_samples).
        Each slice along the last axis is smoothed independently. Entries may be ``-inf`` (zero target density),
        but not ``nan`` or ``+inf``.
    r_eff : float, optional
        Relative efficiency of the draws, i.e., effective sample size divided by the number of draws.
        Default is 1.0, which is appropriate for independent draws, e.g., from an amortized posterior.

    Returns
    -------
    log_weights : np.ndarray
        Pareto-smoothed log weights, normalized to sum to one along the last axis, same shape as ``log_ratios``.
    pareto_k : np.ndarray
        Pareto k diagnostic, shape ``log_ratios.shape[:-1]``. It is ``-inf`` for slices with ratios that are
        constant up to round-off, which need no smoothing. It is ``inf`` or ``nan`` if the tail cannot be fitted,
        e.g., with fewer than about 25 draws or many tied ratios; then the weights are left unsmoothed and must not
        be trusted.

    Raises
    ------
    ImportError
        If ``arviz-stats`` is not installed.
    ValueError
        If ``r_eff`` is not positive, if there are fewer than two draws, if ``log_ratios`` contains ``nan`` or
        ``+inf``, or if all entries of a slice are ``-inf``.
    """
    try:
        from arviz_stats.base import array_stats
    except ImportError as error:
        raise ImportError(
            "Pareto smoothing requires the optional dependency `arviz-stats` (>=1.1.0). "
            "Install it with `pip install arviz-stats`."
        ) from error

    if not r_eff > 0:
        raise ValueError(f"Expected r_eff to be positive, but got {r_eff}.")

    log_ratios = np.asarray(log_ratios, dtype=np.float64)

    if log_ratios.ndim == 0 or log_ratios.shape[-1] < 2:
        raise ValueError(f"Expected at least two draws along the last axis, but got shape {log_ratios.shape}.")

    if np.any(np.isnan(log_ratios) | (log_ratios == np.inf)):
        raise ValueError(
            "log_ratios must not contain nan or +inf. Check that log_q is finite wherever the target density is "
            "positive."
        )

    batch_shape, num_samples = log_ratios.shape[:-1], log_ratios.shape[-1]
    flat_ratios = log_ratios.reshape(-1, num_samples)

    if np.any(np.all(flat_ratios == -np.inf, axis=-1)):
        raise ValueError(
            "All log_ratios of a slice are -inf, i.e., the target density is zero for every draw. Check that the "
            "draws lie within the support of the prior and likelihood."
        )

    # unsmoothed, normalized weights; kept for constant slices and for slices that cannot be smoothed
    log_weights = flat_ratios - logsumexp(flat_ratios, axis=-1, keepdims=True)
    pareto_k = np.full(flat_ratios.shape[0], -np.inf)

    # ratios that are constant up to round-off (e.g., an exact proposal, with log densities possibly computed in
    # float32) need no smoothing, and a tail fitted to round-off errors would give a meaningless k; slices with
    # zero target density for some draws are never treated as constant, since the remaining draws may be few
    scale = np.maximum(1.0, np.max(np.abs(flat_ratios), axis=-1))
    is_constant = np.all(np.isfinite(flat_ratios), axis=-1) & (np.ptp(flat_ratios, axis=-1) <= 1e-6 * scale)
    to_smooth = np.flatnonzero(~is_constant)

    if to_smooth.size == 0:
        return log_weights.reshape(log_ratios.shape), pareto_k.reshape(batch_shape)

    # arviz-stats follows the PSIS-LOO convention and smooths the negated input
    try:
        smoothed, k = array_stats.psislw(-flat_ratios[to_smooth], r_eff=r_eff, axis=-1)
        log_weights[to_smooth] = smoothed
        pareto_k[to_smooth] = k
    except ValueError:
        # some tail cannot be fitted, e.g., too few draws or many tied ratios; smooth the others one by one
        for i in to_smooth:
            try:
                log_weights[i], pareto_k[i] = array_stats.psislw(-flat_ratios[i], r_eff=r_eff, axis=-1)
            except ValueError:
                pareto_k[i] = np.inf

    return log_weights.reshape(log_ratios.shape), pareto_k.reshape(batch_shape)


def psis_correction(
    samples: Mapping[str, np.ndarray],
    log_prior: Callable[[Mapping[str, np.ndarray]], np.ndarray],
    log_likelihood: Callable[[Mapping[str, np.ndarray], Mapping[str, np.ndarray]], np.ndarray],
    *,
    conditions: Mapping[str, np.ndarray],
    log_q: np.ndarray = None,
    estimator: BasicEstimator = None,
    num_resamples: int = None,
    r_eff: float = 1.0,
    seed: int = None,
) -> dict:
    """Corrects approximate posterior draws by Pareto-smoothed importance sampling (PSIS) [1].

    If the likelihood can be evaluated, draws from an amortized posterior approximation q(theta | x) can serve as
    an importance sampling proposal for the true posterior p(theta | x), which is proportional to
    p(theta) p(x | theta). The log importance ratios ``log_prior + log_likelihood - log_q`` are Pareto-smoothed
    with :py:func:`pareto_smooth`. The returned weights correct posterior expectations, and the Pareto k
    diagnostic flags data sets for which the approximation is too poor to be corrected, e.g., due to model
    misspecification.

    Requires the optional dependency ``arviz-stats`` (>=1.1.0), installable with ``pip install arviz-stats``.

    [1] Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024). Pareto smoothed importance sampling.
    Journal of Machine Learning Research, 25(72), 1-58.

    Parameters
    ----------
    samples : dict[str, np.ndarray]
        Approximate posterior draws, with entries of shape (num_datasets, num_samples, ...), e.g., from
        ``estimator.sample(conditions=conditions, num_samples=num_samples)``. For a single data set, entries may
        instead have shape (num_samples, ...) if ``log_q`` is 1-D.
    log_prior : Callable
        ``log_prior(samples)`` returns the (possibly unnormalized) log prior density of the draws,
        shape (num_datasets, num_samples), or (num_samples,) for a single data set.
    log_likelihood : Callable
        ``log_likelihood(samples, conditions)`` returns the log likelihood of the observed data of each data set
        under each draw, shape (num_datasets, num_samples), or (num_samples,) for a single data set.
        Both callables receive ``samples`` (converted to NumPy arrays) and ``conditions`` as passed to this
        function, and return ``-inf`` for draws outside the support.
    conditions : dict[str, np.ndarray]
        Observed data, with entries of shape (num_datasets, ...). Passed to ``log_likelihood``, and to the
        estimator if ``estimator`` is given. For the estimator, entries without a leading axis of length
        num_datasets (e.g., context values that the adapter broadcasts) are passed unchanged.
    log_q : np.ndarray, optional
        Log density of the draws under the approximate posterior, shape (num_datasets, num_samples).
        A 1-D ``log_q`` of shape (num_samples,) denotes a single data set, see ``samples``.
        Exactly one of ``log_q`` and ``estimator`` must be given.
    estimator : BasicEstimator, optional
        Estimator with a density-based inference network that generated the draws. Used to compute ``log_q``
        via :py:meth:`~bayesflow.estimators.BasicEstimator.log_prob`, which includes the Jacobians of the adapter
        transforms, so that ``log_q`` is a density over the same variables as ``log_prior``.
    num_resamples : int, optional
        Number of draws per data set to resample with replacement according to the weights. Defaults to
        num_samples. Set to 0 to skip resampling.
    r_eff : float, optional
        Relative efficiency of the draws, see :py:func:`pareto_smooth`. Default is 1.0.
    seed : int, optional
        Seed for resampling.

    Returns
    -------
    result : dict
        Dictionary with the following entries:

        - ``"log_weights"``: normalized Pareto-smoothed log weights, shape (num_datasets, num_samples).
        - ``"pareto_k"``: Pareto k diagnostic per data set, shape (num_datasets,). A warning is issued for data
          sets with k above ``min(1 - 1 / log10(num_samples), 0.7)`` or with k ``nan`` or ``inf``.
        - ``"ess"``: importance sampling effective sample size per data set, ``r_eff / sum(weights ** 2)``,
          shape (num_datasets,).
        - ``"samples"``: resampled draws with entries of shape (num_datasets, num_resamples, ...),
          only if ``num_resamples > 0``.

        For a single data set (1-D ``log_q``), the leading num_datasets axis is removed from all entries.

    Raises
    ------
    ValueError
        If not exactly one of ``log_q`` and ``estimator`` is given, if ``num_resamples`` is negative, if
        ``samples`` is empty, if the shapes of the samples or log densities do not match
        (num_datasets, num_samples), if ``log_q`` is not finite, if ``log_prior`` or ``log_likelihood`` returns
        ``nan`` or ``+inf``, or if a callable raises a ``KeyError`` for a key of the other argument (e.g., swapped
        ``samples`` and ``conditions``). Also raised for invalid inputs, see :py:func:`pareto_smooth`.

    Examples
    --------
    >>> samples = workflow.sample(conditions=conditions, num_samples=1000)
    >>> result = bf.correction.psis_correction(
    ...     samples, log_prior, log_likelihood, conditions=conditions, estimator=workflow.estimator
    ... )
    >>> result["pareto_k"]  # one value per data set; check before using result["samples"]
    """
    if (log_q is None) == (estimator is None):
        raise ValueError("Exactly one of `log_q` and `estimator` must be given.")

    if num_resamples is not None and num_resamples < 0:
        raise ValueError(f"Expected num_resamples to be non-negative, but got {num_resamples}.")

    samples = {key: np.asarray(keras.ops.convert_to_numpy(value)) for key, value in samples.items()}

    # a 1-D log_q denotes a single data set, whose samples have no leading num_datasets axis
    single_dataset = log_q is not None and np.ndim(keras.ops.convert_to_numpy(log_q)) == 1
    num_leading = 1 if single_dataset else 2
    leading_names = "(num_samples,)" if single_dataset else "(num_datasets, num_samples)"
    if single_dataset:
        hint = (
            " A 1-D log_q denotes a single data set, so samples must have shape (num_samples, ...) and log_prior "
            "and log_likelihood must return shape (num_samples,). For several data sets, pass log_q of shape "
            "(num_datasets, num_samples)."
        )
    elif estimator is not None:
        hint = " For a single data set, add a leading axis of length 1 to all samples and conditions."
    else:
        hint = (
            " For a single data set, pass samples of shape (num_samples, ...) with a 1-D log_q of shape "
            "(num_samples,), or add a leading axis of length 1 to all samples, log_q, and conditions."
        )

    if not samples or any(value.ndim < num_leading for value in samples.values()):
        raise ValueError(
            f"Expected a non-empty dict of samples with entries of leading shape {leading_names}, "
            f"but got shapes { ({key: value.shape for key, value in samples.items()}) }.{hint}"
        )

    leading_shape = next(iter(samples.values())).shape[:num_leading]
    num_datasets, num_samples = (1, *leading_shape) if single_dataset else leading_shape

    if num_datasets == 0:
        raise ValueError("Expected at least one data set, but got num_datasets = 0.")

    if num_samples < 2:
        raise ValueError(f"Expected at least two draws per data set, but got num_samples = {num_samples}.{hint}")

    for key, value in samples.items():
        if value.shape[:num_leading] != leading_shape:
            raise ValueError(
                f"Expected all samples to have leading shape {leading_names} = {leading_shape}, but '{key}' has "
                f"shape {value.shape}. Remove entries that are not posterior draws, e.g., summaries from "
                f"`return_summaries=True`.{hint}"
            )

    if estimator is not None:
        log_q = _estimator_log_q(estimator, samples, conditions, num_datasets, num_samples)

    log_densities = {
        "log_q": log_q,
        "log_prior": _call_log_density("log_prior", log_prior, samples, conditions),
        "log_likelihood": _call_log_density("log_likelihood", log_likelihood, samples, conditions),
    }
    for name, value in log_densities.items():
        value = np.asarray(keras.ops.convert_to_numpy(value), dtype=np.float64)
        if value.shape != leading_shape:
            if value.ndim == len(leading_shape) + 1 and value.shape[:-1] == leading_shape:
                shape_hint = (
                    f" {name} seems to return one value per observation or dimension; sum over the last axis to "
                    "get one value per draw."
                )
            elif value.ndim == 0:
                shape_hint = f" {name} must return one value per draw, not a sum over draws."
            else:
                shape_hint = hint
            raise ValueError(
                f"Expected {name} to have shape {leading_names} = {leading_shape}, but got {value.shape}.{shape_hint}"
            )
        if name == "log_q" and not np.all(np.isfinite(value)):
            num_invalid = int(np.sum(~np.isfinite(value)))
            raise ValueError(
                f"Expected log_q to be finite, but found nan or inf for {num_invalid} of {value.size} draws. "
                "log_q is the log density of the draws under the approximate posterior that generated them."
            )
        num_invalid = int(np.sum(np.isnan(value) | (value == np.inf)))
        if num_invalid > 0:
            raise ValueError(
                f"{name} returned nan or +inf for {num_invalid} of {value.size} draws. "
                "Return -inf for draws outside the support."
            )
        log_densities[name] = value.reshape(num_datasets, num_samples)

    if single_dataset:
        samples = {key: value[None] for key, value in samples.items()}

    log_ratios = log_densities["log_prior"] + log_densities["log_likelihood"] - log_densities["log_q"]
    log_weights, pareto_k = pareto_smooth(log_ratios, r_eff=r_eff)

    k_threshold = min(1 - 1 / np.log10(num_samples), 0.7)
    # nan or inf means the tail could not be fitted, which is unreliable too; -inf (constant ratios) is fine
    num_unreliable = int(np.sum(np.isnan(pareto_k) | (pareto_k > k_threshold)))
    if num_unreliable > 0:
        warnings.warn(
            f"Pareto k exceeds {k_threshold:.2f} for {num_unreliable} of {num_datasets} data sets, so importance "
            "sampling is unreliable there. The approximate posterior is too different from the true posterior, "
            "e.g., due to insufficient training or model misspecification.",
            UserWarning,
            stacklevel=2,
        )

    result = {
        "log_weights": log_weights,
        "pareto_k": pareto_k,
        "ess": r_eff / np.sum(np.exp(2 * log_weights), axis=-1),
    }

    num_resamples = num_samples if num_resamples is None else num_resamples
    if num_resamples > 0:
        # multinomial resampling by inverting the weights' CDF, independently per data set
        cdf = np.cumsum(np.exp(log_weights), axis=-1)
        uniform = np.random.default_rng(seed).random((num_datasets, num_resamples)) * cdf[:, -1:]
        indices = np.stack([np.searchsorted(c, u, side="right") for c, u in zip(cdf, uniform)])
        indices = np.minimum(indices, num_samples - 1)
        result["samples"] = {
            key: np.take_along_axis(value, indices.reshape(indices.shape + (1,) * (value.ndim - 2)), axis=1)
            for key, value in samples.items()
        }

    if single_dataset:
        result = {
            key: {k: v[0] for k, v in value.items()} if key == "samples" else value[0] for key, value in result.items()
        }

    return result


def _call_log_density(
    name: str,
    fn: Callable,
    samples: Mapping[str, np.ndarray],
    conditions: Mapping[str, np.ndarray],
) -> np.ndarray:
    """Calls ``log_prior(samples)`` or ``log_likelihood(samples, conditions)`` and explains failures that are
    likely due to the calling convention, e.g., swapped arguments."""
    call = "log_prior(samples)" if name == "log_prior" else "log_likelihood(samples, conditions)"
    args = (samples,) if name == "log_prior" else (samples, conditions)

    def describe(mapping):
        return {key: getattr(value, "shape", type(value).__name__) for key, value in mapping.items()}

    convention = (
        f"{name} is called as `{call}`, with samples of shapes {describe(samples)} and conditions of shapes "
        f"{describe(conditions)}."
    )

    try:
        return fn(*args)
    except KeyError as error:
        key = error.args[0] if error.args and isinstance(error.args[0], str) else None
        if name == "log_prior" and key in conditions:
            raise ValueError(
                f"log_prior raised KeyError {key!r}, which is a key of conditions. log_prior receives only the "
                f"samples. {convention}"
            ) from error
        if name == "log_likelihood" and (key in samples or key in conditions):
            found = "samples" if key in samples else "conditions"
            raise ValueError(
                f"log_likelihood raised KeyError {key!r}, which is a key of {found}, so its arguments are likely "
                f"swapped. {convention}"
            ) from error
        error.add_note(convention)
        raise
    except Exception as error:
        error.add_note(convention)
        raise


def _estimator_log_q(
    estimator: BasicEstimator,
    samples: Mapping[str, np.ndarray],
    conditions: Mapping[str, np.ndarray],
    num_datasets: int,
    num_samples: int,
) -> np.ndarray:
    """Evaluates the estimator's log density of each draw given its data set, one data set at a time to bound
    memory. Condition entries with a leading axis of length num_datasets are repeated for each draw; other entries,
    e.g., context values that the adapter broadcasts, are passed unchanged."""
    conditions = {key: np.asarray(keras.ops.convert_to_numpy(value)) for key, value in conditions.items()}

    def is_per_dataset(value):
        return value.ndim > 0 and value.shape[0] == num_datasets

    log_q = np.empty((num_datasets, num_samples))
    for j in range(num_datasets):
        data = {
            key: np.repeat(value[j : j + 1], num_samples, axis=0) if is_per_dataset(value) else value
            for key, value in conditions.items()
        }
        data |= {key: value[j] for key, value in samples.items()}
        log_q[j] = np.reshape(estimator.log_prob(data), (num_samples,))

    return log_q
