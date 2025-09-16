import numpy as np
from scipy import special
from collections.abc import Sequence


def inverse_sigmoid(x: np.ndarray) -> np.ndarray:
    """Inverse of the sigmoid function."""
    return np.log(x) - np.log1p(-x)


def inverse_shifted_softplus(
    x: np.ndarray, shift: float = np.log(np.e - 1), beta: float = 1.0, threshold: float = 20.0
) -> np.ndarray:
    """Inverse of the shifted softplus function."""
    return inverse_softplus(x, beta=beta, threshold=threshold) - shift


def inverse_softplus(x: np.ndarray, beta: float = 1.0, threshold: float = 20.0) -> np.ndarray:
    """Numerically stabilized inverse softplus function."""
    with np.errstate(over="ignore"):
        expm1_x = np.expm1(x)
    return np.where(beta * x > threshold, x, np.log(beta * expm1_x) / beta)


def one_hot(indices: np.ndarray, num_classes: int, dtype: str = "float32") -> np.ndarray:
    """Converts a 1D array of indices to a one-hot encoded 2D array."""
    return np.eye(num_classes, dtype=dtype)[indices]


def shifted_softplus(
    x: np.ndarray, beta: float = 1.0, threshold: float = 20.0, shift: float = np.log(np.e - 1)
) -> np.ndarray:
    """Shifted version of the softplus function such that shifted_softplus(0) = 1"""
    return softplus(x + shift, beta=beta, threshold=threshold)


sigmoid = special.expit
softmax = special.softmax


def softplus(x: np.ndarray, beta: float = 1.0, threshold: float = 20.0) -> np.ndarray:
    """Numerically stabilized softplus function."""
    with np.errstate(over="ignore"):
        exp_beta_x = np.exp(beta * x)
    return np.where(beta * x > threshold, x, np.log1p(exp_beta_x) / beta)


def credible_interval(x: np.ndarray, prob: float = 0.95, axis: Sequence[int] | int = None, **kwargs) -> np.ndarray:
    """
    Compute credible interval from samples using quantiles.

    Parameters
    ----------
    x : array_like
        Input array of samples from a posterior distribution or bootstrap samples.
    prob : float, default 0.95
        Coverage probability of the credible interval (between 0 and 1).
        For example, 0.95 gives a 95% credible interval.
    axis : Sequence[int]
        Axis or axes along which the credible interval is computed.
        Default is None (flatten array).

    Returns
    -------
    a numpy array of shape (2, ...) with the first dimension indicating the
    lower and upper bounds of the credible interval.

    Examples
    --------
    >>> import numpy as np
    >>> # Simulate posterior samples
    >>> samples = np.random.normal(size=(10, 1000, 3))

    >>> # Different coverage probabilities
    >>> credible_interval(samples, prob=0.5, axis=1)  # 50% CI
    >>> credible_interval(samples, prob=0.99, axis=1)  # 99% CI
    """

    # Input validation
    if not 0 <= prob <= 1:
        raise ValueError(f"prob must be between 0 and 1, got {prob}")

    # Calculate tail probabilities
    alpha = 1 - prob
    lower_q = alpha / 2
    upper_q = 1 - alpha / 2

    # Compute quantiles
    return np.quantile(x, q=(lower_q, upper_q), axis=axis, **kwargs)
