import numpy as np

from .benchmark_simulator import BenchmarkSimulator


class SLCP(BenchmarkSimulator):
    def __init__(
        self,
        lower_bound: float = -3.0,
        upper_bound: float = 3.0,
        n_obs: int = 4,
        flatten: bool = True,
        rng: np.random.Generator = None,
    ):
        """SLCP simulated benchmark
        See https://arxiv.org/pdf/2101.04653.pdf, Task T.3

        Parameters
        ----------
        lower_bound: float, optional, default: -3.0
            The lower bound of the uniform prior.
        upper_bound: float, optional, default: 3.0
            The upper bound of the uniform prior.
        n_obs: int, optional, default: 4
            The number of observations to generate from the slcp likelihood.
        flatten: bool, optional, default: True
            A flag to indicate whather a 1D (`flatten=True`) or a 2D (`flatten=False`)
            representation of the simulated data is returned.
        rng: np.random.Generator or None, optional, default: None
            An optional random number generator to use.
        """

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_obs = n_obs
        self.flatten = flatten
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

    def prior(self):
        """Generates a random draw from a 5-dimensional uniform prior bounded between
        `lower_bound` and `upper_bound` which represents the 5 parameters of the SLCP
        simulator.

        Returns
        -------
        params : np.ndarray of shape (5, )
            A single draw from the 5-dimensional uniform prior.
        """

        return self.rng.uniform(low=self.lower_bound, high=self.upper_bound, size=5)

    def observation_model(self, params: np.ndarray):
        """Generates data from the SLCP model designed as a benchmark for a simple likelihood
        and a complex posterior due to a non-linear pushforward params -> x.

        Parameters
        ----------
        params  : np.ndarray of shape (params, D)
            The location parameters of the Gaussian likelihood.

        Returns
        -------
        x : np.ndarray of shape (n_obs*2, ) or (n_obs, 2), as indictated by the `flatten`
            boolean flag. The sample of simulated data from the SLCP model.
        """

        # Specify 2D location
        loc = np.array([params[0], params[1]])

        # Specify 2D covariance matrix
        s1 = params[2] ** 2
        s2 = params[3] ** 2
        rho = np.tanh(params[4])
        cov = rho * s1 * s2
        S_param = np.array([[s1**2, cov], [cov, s2**2]])

        # Obtain given number of draws from the MVN likelihood
        x = self.rng.multivariate_normal(loc, S_param, size=self.n_obs)
        if self.flatten:
            return x.flatten()
        return x
