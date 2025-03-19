import numpy as np
from scipy.integrate import odeint

from .benchmark_simulator import BenchmarkSimulator


class LotkaVolterra(BenchmarkSimulator):
    def __init__(
        self,
        X0: int = 30,
        Y0: int = 1,
        T: int | None = 20,
        subsample: int = 10,
        flatten: bool = True,
        obs_noise: float = 0.1,
        dt: float = None,
        rng: np.random.Generator = None,
    ):
        """Lotka Volterra simulated benchmark.
        See: https://arxiv.org/pdf/2101.04653.pdf, Task T.10

        Parameters
        ----------
        X0: int, optional, default: 30
            Initial number of prey species.
        Y0: int, optional, default: 1
            Initial number of predator species.
        T: int, optional, default: 20
            The duration (time horizon) of the simulation.
        subsample: int or None, optional, default: 10
            The number of evenly spaced time points to return.
            If None, no subsampling will be performed and all T timepoints will be returned.
        flatten: bool, optional, default: True
            A flag to indicate whether a 1D (`flatten=True`) or 2D (`flatten=False`)
            representation of the simulated data is returned.
        obs_noise: float, optional, default: 0.1
            The standard deviation of the log-normal likelihood.
        rng: np.random.Generator or None, optional, default: None
            An optional random number generator to use.
        """

        self.X0 = X0
        self.Y0 = Y0
        self.T = T
        if dt is None:
            dt = 1 / T
        self.dt = dt
        self.subsample = subsample
        self.flatten = flatten
        self.obs_noise = obs_noise
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

    def _deriv(self, x, t, alpha, beta, gamma, delta):
        """Helper function for scipy.integrate.odeint."""

        X, Y = x
        dX = alpha * X - beta * X * Y
        dY = -gamma * Y + delta * X * Y
        return dX, dY

    def prior(self):
        """Generates a random draw from a 4-dimensional (independent) lognormal prior
        which represents the four contact parameters of the Lotka-Volterra model.

        Returns
        -------
        params : np.ndarray of shape (4, )
            A single draw from the 4-dimensional prior.
        """

        params = self.rng.lognormal(mean=[-0.125, -3, -0.125, -3], sigma=0.5)
        return params

    def observation_model(self, params: np.ndarray) -> np.ndarray:
        """Runs a Lotka-Volterra simulation for T time steps and returns `subsample` evenly spaced
        points from the simulated trajectory, given contact parameters `params`.

        Parameters
        ----------
        params      : np.ndarray of shape (2,)
            The 2-dimensional vector of disease parameters.

        Returns
        -------
        x : np.ndarray of shape (subsample, 2) or (subsample*2,) if `subsample is not None`,
            otherwise shape (T, 2) or (T*2,) if `subsample is None`.
            The time series of simulated predator and pray populations
        """

        # Create vector (list) of initial conditions
        x0 = self.X0, self.Y0

        # Unpack parameter vector into scalars
        alpha, beta, gamma, delta = params

        # Prepate time vector between 0 and T of length T
        t_vec = np.linspace(0, self.T, int(1 / self.dt))

        # Integrate using scipy and retain only infected (2-nd dimension)
        pp = odeint(self._deriv, x0, t_vec, args=(alpha, beta, gamma, delta))

        # Subsample evenly the specified number of points, if specified
        if self.subsample is not None:
            pp = pp[:: (self.T // self.subsample)]

        # Ensure minimum count is 0, which will later pass by log(0 + 1)
        pp[pp < 0] = 0.0

        # Add noise, decide whether to flatten and return
        x = self.rng.lognormal(np.log1p(pp), sigma=self.obs_noise)
        if self.flatten:
            return x.flatten()
        return x
