import numpy as np
from scipy.stats import multivariate_t

from .benchmark_simulator import BenchmarkSimulator


class SLCPDistractors(BenchmarkSimulator):
    def __init__(
        self,
        lower_bound: float = -3.0,
        upper_bound: float = 3.0,
        n_obs: int = 4,
        n_dist: int = 46,
        dim: int = 2,
        mu_scale: float = 15.0,
        shape_scale: float = 0.01,
        flatten: bool = True,
        rng: np.random.Generator = None,
    ):
        """SLCP Distractors simulated benchmark
        See: https://arxiv.org/pdf/2101.04653.pdf, Task T.4

        Parameters
        ----------
        lower_bound: float, optional, default: -3.0
            The lower bound of the uniform prior.
        upper_bound: float, optional, default: 3.0
            The upper bound of the uniform prior.
        n_obs: int, optional, default: 4
            The number of observations to generate from the slcp likelihood.
        n_dist: int, optional, default: 46
            The number of distractor to draw from the distractor likelihood.
        dim: int, optional, default: 2
            The dimensionality of each student-t distribution in the mixture.
        mu_scale: float, optional, default: 15.0
            The scale of the zero-centered Gaussian prior from which the mean vector
            of each student-t distribution in the mixture is drawn.
        shape_scale: float, optional, default: 0.01
            The scale of the assumed `np.eye(dim)` shape matrix. The default is chosen to keep
            the scale of the distractors and observations relatively similar.
        flatten: bool, optional, default: True
            A flag to indicate whather a 1D (`flatten=True`) or a 2D (`flatten=False`)
            representation of the simulated data is returned.
        rng: np.random.Generator or None, optional, default: None
            An optional random number generator to use.
        """

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.n_obs = n_obs
        self.n_dist = n_dist
        self.dim = dim
        self.mu_scale = mu_scale
        self.shape_scale = shape_scale
        self.flatten = flatten
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

    def _get_random_student_t(self):
        """A helper function to create a "frozen" multivariate student-t distribution of dimensions `dim`.

        Returns
        -------
        student : callable (scipy.stats._multivariate.multivariate_t_frozen)
            The student-t generator.
        """

        # Draw mean and return student-t object
        mu = self.mu_scale * self.rng.normal(size=self.dim)
        return multivariate_t(loc=mu, shape=self.shape_scale, df=2, allow_singular=True, seed=self.rng)

    def _draw_mixture_student_t(self, num_students: int):
        """Helper function to generate `n_draws` random draws from a mixture of `num_students`
        multivariate Student-t distributions.

        Uses the function `get_random_student_t` to create each of the studen-t callable objects.

        Parameters
        ----------
        num_students : int
            The number of multivariate student-t mixture components

        Returns
        -------
        sample : np.ndarray of shape (n_draws, dim)
            The random draws from the mixture of students.
        """

        # Obtain a list of scipy frozen distributions (each will have a different mean)
        students = [self._get_random_student_t() for _ in range(num_students)]

        # Obtain the sample of n_draws from the mixture and return
        sample = [students[self.rng.integers(low=0, high=num_students)].rvs() for _ in range(self.n_dist)]
        return np.array(sample)

    def prior(self):
        """Generates a random draw from a 5-dimensional uniform prior bounded between
        `lower_bound` and `upper_bound`.

        Returns
        -------
        params : np.ndarray of shape (5, )
            A single draw from the 5-dimensional uniform prior.
        """

        return self.rng.uniform(low=self.lower_bound, high=self.upper_bound, size=5)

    def observation_model(self, params: np.ndarray):
        """Generates data from the SLCP model designed as a benchmark for a simple likelihood
        and a complex posterior due to a non-linear pushforward params -> x. In addition, it
        outputs uninformative distractor data.

        Parameters
        ----------
        params: np.ndarray of shape (params, D)
            The location parameters of the Gaussian likelihood.

        Returns
        -------
        x: np.ndarray of shape (n_obs*2 + n_dist*2, ) if `flatten=True`, otherwise
            np.ndarray of shape (n_obs + n_dist, 2) if `flatten=False`
        """

        # Specify 2D location
        loc = np.array([params[0], params[1]])

        # Specify 2D covariance matrix
        s1 = params[2] ** 2
        s2 = params[3] ** 2
        rho = np.tanh(params[4])
        cov = rho * s1 * s2
        S_param = np.array([[s1**2, cov], [cov, s2**2]])

        # Obtain informative part of the data
        x_info = self.rng.multivariate_normal(loc, S_param, size=self.n_obs)

        # Obtain uninformative part of the data
        x_uninfo = self._draw_mixture_student_t(20)

        # Concatenate informative with uninformative and return
        x = np.concatenate([x_info, x_uninfo], axis=0)
        if self.flatten:
            return x.flatten()
        return x
