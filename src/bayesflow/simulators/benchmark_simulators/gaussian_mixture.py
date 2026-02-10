import numpy as np

from .benchmark_simulator import BenchmarkSimulator


class GaussianMixture(BenchmarkSimulator):
    def __init__(
        self,
        D: int = 2,
        lower_bound: float = -10.0,
        upper_bound: float = 10.0,
        prob: float = 0.5,
        scale_c1: float = 1.0,
        scale_c2: float = 0.1,
        rng: np.random.Generator = None,
    ):
        """Gaussian Mixture simulated benchmark
        See: https://arxiv.org/pdf/2101.04653.pdf, Task T.7

        Important: The parameterization uses scales, so use sqrt(var),
        if you want to be working with variances instead of scales.

        Parameters
        ----------
        D: int, optional, default: 2
            The dimensionality of the mixture model.
        lower_bound: float, optional, default: -10.0
            The lower bound of the uniform prior.
        upper_bound: float, optional, default: 10.0
            The upper bound of the uniform prior.
        prob: float, optional, default: 0.5
            The mixture probability (coefficient).
        scale_c1: float, optional, default: 1.0
            The scale of the first component.
        scale_c2: float, optional, default: 0.1
            The scale of the second component.
        rng: np.random.Generator or None, optional, default: None
            An optional random number generator to use.
        """

        self.D = D
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.prob = prob
        self.scale_c1 = scale_c1
        self.scale_c2 = scale_c2
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

    def prior(self):
        """Generates a random draw from a 2-dimensional uniform prior bounded between
        `lower_bound` and `upper_bound` representing the common mean of a 2D Gaussian
        mixture model (GMM).

        Returns
        -------
        params : np.ndarray of shape (D, )
            A single draw from the D-dimensional uniform prior
        """

        return self.rng.uniform(low=self.lower_bound, high=self.upper_bound, size=self.D)

    def observation_model(self, params: np.ndarray):
        """Simulates data from the Gaussian mixture model (GMM) with
        shared location vector. For more details, see

        Parameters
        ----------
        params   : np.ndarray of shape (D, )
            The D-dimensional vector of parameter locations.

        Returns
        -------
        x : np.ndarray of shape (2, )
            The 2D vector generated from the GMM simulator.
        """

        # Draw component index
        idx = self.rng.binomial(n=1, p=self.prob)

        # Draw 2D-Gaussian sample according to component index
        if idx == 0:
            return self.rng.normal(loc=params, scale=self.scale_c1)
        return self.rng.normal(loc=params, scale=self.scale_c2)
