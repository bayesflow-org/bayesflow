import numpy as np
from scipy.special import expit

from .benchmark_simulator import BenchmarkSimulator


class BernoulliGLM(BenchmarkSimulator):
    def __init__(self, T: int = 100, scale_by_T: bool = True, rng: np.random.Generator = None):
        """Bernoulli GLM simulated benchmark
        See: https://arxiv.org/pdf/2101.04653.pdf, Task T.5

        Important: `scale_sum` should be set to False if the simulator is used
        with variable `T` during training, otherwise the information of `T` will
        be lost.

        Parameters
        ----------
        T: int, optional, default: 100
            The simulated duration of the task (eq. the number of Bernoulli draws).
        scale_by_T: bool, optional, default: True
            A flag indicating whether to scale the summayr statistics by T.
        rng: np.random.Generator or None, optional, default: None
            An optional random number generator to use.
        """

        self.T = T
        self.scale_by_T = scale_by_T
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

        # Covariance matrix computed once for efficiency
        F = np.zeros((9, 9))
        i = np.arange(9)
        F[i, i] = 1 + np.sqrt(i / 9)
        F[i[1:], i[:-1]] = -2
        F[i[2:], i[:-2]] = 1
        self.cov = np.linalg.inv(F.T @ F)

    def prior(self):
        """Generates a random draw from the custom prior over the 10
        Bernoulli GLM parameters (1 intercept and 9 weights). Uses a
        global covariance matrix `Cov` for the multivariate Gaussian prior
        over the model weights, which is pre-computed for efficiency.

        Returns
        -------
        params: np.ndarray of shape (10, )
            A single draw from the prior.
        """

        beta = self.rng.normal(0, 2)
        f = self.rng.multivariate_normal(np.zeros(9), self.cov)
        return np.append(beta, f)

    def observation_model(self, params: np.ndarray):
        """Simulates data from the custom Bernoulli GLM likelihood.

        Parameters
        ----------
        params: np.ndarray of shape (10, )
            The vector of model parameters (`params[0]` is intercept, `params[i], i > 0` are weights).

        Returns
        -------
        x: np.ndarray of shape (10, )
            The vector of sufficient summary statistics of the data.
        """

        # Unpack parameters
        beta, f = params[0], params[1:]

        # Generate design matrix
        V = self.rng.normal(size=(9, self.T))

        # Draw from Bernoulli GLM
        z = self.rng.binomial(n=1, p=expit(V.T @ f + beta))

        # Compute and return (scaled) sufficient summary statistics
        x1 = np.sum(z)
        x_rest = V @ z
        x = np.append(x1, x_rest)
        if self.scale_by_T:
            x /= self.T
        return x
