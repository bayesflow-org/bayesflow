from collections.abc import Mapping, Sequence

import numpy as np


def _inverse_square_root_covariance(covariance: np.ndarray, ridge: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(covariance + ridge * np.eye(covariance.shape[0]))
    return (eigenvectors / np.sqrt(eigenvalues)) @ eigenvectors.T


def canonical_correlation_metric(
    summaries: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    *,
    summary_keys: Sequence[str] | str | None = None,
    target_keys: Sequence[str] | str | None = None,
    ridge: float = 1e-8,
) -> dict[str, np.ndarray | str | list[str]]:
    """Compute canonical correlations between summaries and target features.

    Dictionary inputs are flattened per key and concatenated along the feature
    axis. Array inputs are flattened after the leading dataset axis.

    This is useful for checking whether learned summaries preserve the
    information directions contained in chosen target features. For summaries
    ``S`` and targets ``T``, the returned values are the singular values of
    ``Cov(S)^(-1/2) Cov(S, T) Cov(T)^(-1/2)``. Equivalently, each value is the
    maximum correlation between a linear projection of ``S`` and a linear
    projection of ``T``, constrained to be orthogonal to earlier canonical
    directions. If all target-dimensional values are close to one, then the
    target feature vector is linearly recoverable from the summaries up to an
    invertible change of coordinates; small values indicate target directions
    that the summaries have collapsed or failed to expose.

    The diagnostic is most interpretable for unimodal or sufficient-statistic
    settings where the chosen targets (e.g., posterior means, parameters, or
    analytic sufficient statistics), are good single-vector summaries of
    the posterior. For multimodal or symmetry-heavy posteriors, low or high
    canonical correlations with one target vector need not imply poor or good
    posterior calibration.

    Examples
    --------
    >>> summaries = {"summary": np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]])}
    >>> simulations = {"x": np.array([[[0.0, 1.0]], [[1.0, 0.0]], [[2.0, 1.0]]])}
    >>> out = canonical_correlation_metric(summaries, simulations, summary_keys="summary", target_keys="x")
    >>> out["values"].round(3)
    array([1., 1.])
    >>> out["variable_names"]
    ['canonical_correlation_1', 'canonical_correlation_2']
    """

    if isinstance(summaries, Mapping):
        summary_keys = (summary_keys,) if isinstance(summary_keys, str) else summary_keys or summaries
        summaries = np.concatenate(
            [summaries[key].reshape(summaries[key].shape[0], -1) for key in summary_keys],
            axis=-1,
        )
    else:
        summaries = summaries.reshape(summaries.shape[0], -1)

    if isinstance(targets, Mapping):
        target_keys = (target_keys,) if isinstance(target_keys, str) else target_keys or targets
        targets = np.concatenate(
            [targets[key].reshape(targets[key].shape[0], -1) for key in target_keys],
            axis=-1,
        )
    else:
        targets = targets.reshape(targets.shape[0], -1)

    if summaries.shape[0] != targets.shape[0]:
        raise ValueError("'summaries' and 'targets' must have the same number of datasets.")

    summaries = summaries - np.mean(summaries, axis=0, keepdims=True)
    targets = targets - np.mean(targets, axis=0, keepdims=True)

    normalizer = summaries.shape[0] - 1
    summaries_covariance = (summaries.T @ summaries) / normalizer
    targets_covariance = (targets.T @ targets) / normalizer
    cross_covariance = (summaries.T @ targets) / normalizer

    values = np.linalg.svd(
        _inverse_square_root_covariance(summaries_covariance, ridge)
        @ cross_covariance
        @ _inverse_square_root_covariance(targets_covariance, ridge),
        compute_uv=False,
    )

    return {
        "values": values,
        "metric_name": "Canonical Correlation Metric",
        "variable_names": [f"canonical_correlation_{i + 1}" for i in range(values.shape[0])],
    }
