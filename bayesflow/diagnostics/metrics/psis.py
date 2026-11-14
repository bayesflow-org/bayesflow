import numpy as np

from bayesflow.utils import logging


def pareto_smooth(log_ratios: np.ndarray, *, r_eff: float = 1.0) -> dict[str, np.ndarray]:
    """Pareto-smooths importance ratios and computes the Pareto k diagnostic [1].

    To correct approximate posterior samples, use ``log_ratios = log_prior + log_likelihood - log_posterior``,
    where ``log_posterior`` is the log density of the approximate posterior that generated the samples.
    A warning is issued if the Pareto k diagnostic indicates unreliable weights.

    [1] Vehtari, A., Simpson, D., Gelman, A., Yao, Y., & Gabry, J. (2024). Pareto smoothed importance sampling.
    Journal of Machine Learning Research, 25(72), 1-58.

    Parameters
    ----------
    log_ratios : np.ndarray
        Log importance ratios, shape (..., num_samples). Each slice along the last axis is smoothed independently.
    r_eff : float, optional
        Relative efficiency of the samples, i.e., effective sample size divided by num_samples.
        Default is 1.0, which is appropriate for independent samples.

    Returns
    -------
    result : dict
        Dictionary containing:

        - "log_weights" : np.ndarray
            Normalized smoothed log weights, shape (..., num_samples).
        - "pareto_k" : np.ndarray
            Pareto k diagnostic, shape (...). Importance sampling is unreliable for values above
            ``min(1 - 1 / log10(num_samples), 0.7)``.

    Examples
    --------
    Correct the samples of a trained ``workflow`` for the model theta ~ N(0, 1), x_i ~ N(theta, 1), given one observed
    data set ``conditions`` (leading axis of size 1). ``log_prob`` takes one copy of the conditions per sample:

    >>> samples = workflow.sample(conditions=conditions, num_samples=1000)
    >>> theta = samples["theta"][0]
    >>> repeated = {key: np.repeat(value, 1000, axis=0) for key, value in conditions.items()}
    >>> log_posterior = workflow.log_prob(repeated | {"theta": theta})
    >>> log_prior = scipy.stats.norm.logpdf(theta[:, 0], 0.0, 1.0)
    >>> log_likelihood = scipy.stats.norm.logpdf(conditions["x"][0], theta, 1.0).sum(axis=-1)
    >>> result = bf.diagnostics.pareto_smooth(log_prior + log_likelihood - log_posterior)
    """
    try:
        from arviz_stats.base import array_stats
    except ImportError as e:
        raise ImportError("Could not import arviz_stats. Please install it via `pip install arviz-stats`") from e

    # arviz-stats follows the PSIS-LOO convention and smooths the negated input
    log_weights, pareto_k = array_stats.psislw(-np.asarray(log_ratios), r_eff=r_eff, axis=-1)

    threshold = min(1 - 1 / np.log10(np.shape(log_ratios)[-1]), 0.7)
    if not np.all(pareto_k <= threshold):
        logging.warning(f"Pareto k exceeds {threshold:.2f} or is nan, so the importance weights may be unreliable.")

    return {"log_weights": log_weights, "pareto_k": pareto_k}
