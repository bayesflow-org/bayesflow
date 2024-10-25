import numpy as np

from .simulator import Simulator
from ..types import Shape


class TwoMoons(Simulator):
    def __init__(self, lower_bound: float = -1.0, upper_bound: float = 1.0, seed: int = None):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.rng = np.random.default_rng(seed)

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        r = self.rng.normal(0.1, 0.01, size=batch_shape + (1,))
        alpha = self.rng.uniform(-0.5 * np.pi, 0.5 * np.pi, size=batch_shape + (1,))
        theta = self.rng.uniform(self.lower_bound, self.upper_bound, size=batch_shape + (2,))

        x1 = -np.abs(theta[..., :1] + theta[..., 1:]) / np.sqrt(2.0) + r * np.cos(alpha) + 0.25
        x2 = (-theta[..., :1] + theta[..., 1:]) / np.sqrt(2.0) + r * np.sin(alpha)

        x = np.concatenate([x1, x2], axis=-1)

        return dict(r=r, alpha=alpha, theta=theta, x=x)
