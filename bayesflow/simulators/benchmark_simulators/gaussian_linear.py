import numpy as np

from .benchmark_simulator import BenchmarkSimulator


class GaussianLinear(BenchmarkSimulator):
    def __init__(
        self,
        D: int = 10,
        prior_scale: float = 0.1,
        n_obs: int = None,
        obs_scale: float = 0.1,
        rng: np.random.Generator = None,
    ):
        """Gaussian Linear simulated benchmark
        See: https://arxiv.org/pdf/2101.04653.pdf, Task T.1

        NOTE: The paper description uses a variance of 0.1 for the prior and likelihood
        but the implementation uses scale = 0.1 Our implmenetation uses a default scale
        of 0.1 for consistency with the implementation.

        Parameters
        ----------
        D: int, optional, default: 10
            The dimensionality of the Gaussian prior distribution.
        prior_scale: float, optional, default: 0.1
            The scale of the Gaussian prior.
        n_obs: int or None, optional, default: None
            The number of observations to draw from the likelihood given the location
            parameter `params`. If `n obs is None`, a single draw is produced.
        obs_scale: float, optional, default: 0.1
            The scale of the Gaussian likelihood.
        rng: np.random.Generator or None, optional, default: None
            An optional random number generator to use.
        """

        self.D = D
        self.prior_scale = prior_scale
        self.n_obs = n_obs
        self.obs_scale = obs_scale
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

    def prior(self):
        """Generates a random draw from a D-dimensional Gaussian prior distribution with a
        spherical scale matrix given by sigma * I_D. Represents the location vector of
        a (conjugate) Gaussian likelihood.

        Returns
        -------
        params : np.ndarray of shape (D, )
            A single draw from the D-dimensional Gaussian prior.
        """

        return self.prior_scale * self.rng.normal(size=self.D)

    def observation_model(self, params: np.ndarray):
        """Generates batched draws from a D-dimenional Gaussian distributions given a batch of
        location (mean) parameters of D dimensions. Assumes a spherical convariance matrix given
        by scale * I_D.

        Parameters
        ----------
        params : np.ndarray of shape (params, D)
            The location parameters of the Gaussian likelihood.

        Returns
        -------
        x : np.ndarray of shape (params.shape[0], params.shape[1]) if n_obs is None,
            else np.ndarray of shape (params.shape[0], n_obs, params.shape[1])
            A single draw or a sample from a batch of Gaussians.
        """

        # Generate prior predictive samples, possibly a single if n_obs is None
        if self.n_obs is None:
            return self.rng.normal(loc=params, scale=self.obs_scale)
        if params.ndim == 2:
            # batched sampling with n_obs
            x = self.rng.normal(loc=params, scale=self.obs_scale, size=(self.n_obs, params.shape[0], params.shape[1]))
            return np.transpose(x, (1, 0, 2))
        elif params.ndim == 1:
            # non-batched sampling with n_obs
            return self.rng.normal(loc=params, scale=self.obs_scale, size=(self.n_obs, params.shape[0]))
