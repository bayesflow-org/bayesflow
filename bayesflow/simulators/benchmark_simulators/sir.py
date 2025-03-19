import numpy as np
from scipy.integrate import odeint

from .benchmark_simulator import BenchmarkSimulator


class SIR(BenchmarkSimulator):
    def __init__(
        self,
        N: float = 1e6,
        T: int = 160,
        I0: float = 1.0,
        R0: float = 0.0,
        subsample: int = None,
        total_count: int = 1000,
        scale_by_total: bool = True,
        rng: np.random.Generator = None,
    ):
        """SIR simulated benchmark
        See: https://arxiv.org/pdf/2101.04653.pdf, Task T.9

        NOTE: the simulator scales outputs between 0 and 1.

        Parameters
        ----------
        N: float, optional, default: 1e6
            The size of the simulated population.
        T: int, optional, default: 160
            The duration (time horizon) of the simulation.
        I0: float, optional, default: 1.0
            The number of initially infected individuals.
        R0: float, optional, default: 0.0
            The number of initially recovered individuals.
        subsample: int or None, optional, default: 10
            The number of evenly spaced time points to return. If `None`,
            no subsampling will be performed and all `T` timepoints will be returned.
        total_count: int, optional, default: 1000
            The `N` parameter of the binomial noise distribution. Used just
            for scaling the data and magnifying the effect of noise, such that
            `max infected == total_count`.
        scale_by_total: bool, optional, default: True
            Scales the outputs by `total_count` if set to True.
        rng: np.random.Generator or None, optional, default: None
            An optional random number generator to use.
        """

        self.N = N
        self.T = T
        self.I0 = I0
        self.R0 = R0
        self.subsample = subsample
        self.total_count = total_count
        self.scale_by_total = scale_by_total
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

    def _deriv(self, x, t, N, beta, gamma):
        """Helper function for scipy.integrate.odeint."""

        s, i, r = x
        dS = -beta * s * i / N
        dI = beta * s * i / N - gamma * i
        dR = gamma * i
        return dS, dI, dR

    def prior(self):
        """Generates a random draw from a 2-dimensional (independent) lognormal prior
        which represents the contact and recovery rate parameters of a basic SIR model.

        Returns
        -------
        params : np.ndarray of shape (2, )
            A single draw from the 2-dimensional prior.
        """

        return self.rng.lognormal(mean=[np.log(0.4), np.log(1 / 8)], sigma=[0.5, 0.2])

    def observation_model(self, params: np.ndarray):
        """Runs a basic SIR model simulation for T time steps and returns `subsample` evenly spaced
        points from the simulated trajectory, given disease parameters (contact and recovery rate) `params`.

        Parameters
        ----------
        params         : np.ndarray of shape (2,)
            The 2-dimensional vector of disease parameters.

        Returns
        -------
        x : np.ndarray of shape (subsample, ) or (T, 1) if subsample=None
            The time series of simulated infected individuals. A trailing dimension of 1 should
            be added by a BayesFlow configurator if the data is (properly) to be treated as time series.
        """

        # Create vector (list) of initial conditions
        x0 = self.N - self.I0 - self.R0, self.I0, self.R0

        # Unpack parameter vector into scalars
        beta, gamma = params

        # Prepate time vector between 0 and T of length T
        t_vec = np.linspace(0, self.T, self.T)

        # Integrate using scipy and retain only infected (2-nd dimension)
        irt = odeint(self._deriv, x0, t_vec, args=(self.N, beta, gamma))[:, 1]

        # Subsample evenly the specified number of points, if specified
        if self.subsample is not None:
            irt = irt[:: (self.T // self.subsample)]
        else:
            irt = irt[:, None]

        # Truncate irt, so that small underflow below zero becomes zero
        irt = np.maximum(irt, 0.0)

        # Add noise and scale, if indicated
        x = self.rng.binomial(n=self.total_count, p=irt / self.N)
        if self.scale_by_total:
            x = x / self.total_count

        return x
