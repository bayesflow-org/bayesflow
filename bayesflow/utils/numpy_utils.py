import numpy as np
from scipy import special


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
