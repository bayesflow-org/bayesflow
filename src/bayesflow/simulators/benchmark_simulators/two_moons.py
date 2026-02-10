import numpy as np

from .benchmark_simulator import BenchmarkSimulator


class TwoMoons(BenchmarkSimulator):
    def __init__(self, lower_bound: float = -1.0, upper_bound: float = 1.0, rng: np.random.Generator = None):
        """Two moons simulated benchmark.
        See: https://arxiv.org/pdf/2101.04653.pdf, Task T.8

        Parameters
        ----------
        lower_bound: float, optional, default: -1.0
            The lower bound of the uniform prior
        upper_bound: float, optional, default:  1.0
            The upper bound of the uniform prior
        rng: np.random.Generator or None, optional, default: None
            An option random number generator to use
        """

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

    def prior(self):
        """Generates a random draw from a 2-dimensional uniform prior bounded between
        `lower_bound` and `upper_bound` which represents the two parameters of the two moons simulator.

        Returns
        -------
        params: np.ndarray of shape (2, )
            A single draw from the 2-dimensional uniform prior.
        """

        return self.rng.uniform(low=self.lower_bound, high=self.upper_bound, size=2)

    def observation_model(self, params: np.ndarray):
        """Implements data generation from the two-moons model with a bimodal posterior.

        Parameters
        ----------
        params: np.ndarray of shape (2, )
            The vector of two model parameters.

        Returns
        -------
        observables: np.ndarray of shape (2, )
            The 2D vector generated from the two moons simulator.
        """

        # Generate noise
        alpha = self.rng.uniform(low=-0.5 * np.pi, high=0.5 * np.pi)
        r = self.rng.normal(loc=0.1, scale=0.01)

        # Forward process
        rhs1 = np.array([r * np.cos(alpha) + 0.25, r * np.sin(alpha)])
        rhs2 = np.array([-np.abs(params[0] + params[1]) / np.sqrt(2.0), (-params[0] + params[1]) / np.sqrt(2.0)])

        return rhs1 + rhs2
