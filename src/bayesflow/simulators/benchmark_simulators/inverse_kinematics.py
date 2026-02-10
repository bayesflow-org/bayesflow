import numpy as np

from .benchmark_simulator import BenchmarkSimulator


class InverseKinematics(BenchmarkSimulator):
    def __init__(
        self,
        scales: np.ndarray = None,
        l1: float = 0.5,
        l2: float = 0.5,
        l3: float = 1.0,
        rng: np.random.Generator = None,
    ):
        """Inverse Kinematics simulated benchmark
        See: https://arxiv.org/pdf/2101.10763.pdf

        Parameters
        ----------
        scales: np.ndarray of shape (4, ) or None, optional, default: None
            The four scales of the Gaussian prior.
            If ``None`` provided, the scales from https://arxiv.org/pdf/2101.10763.pdf
            will be used: [0.25, 0.5, 0.5, 0.5].
        l1: float, optional, default: 0.5
            The length of the first segment.
        l2: float, optional, default: 0.5
            The length of the second segment.
        l3: float, optional, default: 1.0
            The length of the third segment.
        rng: np.random.Generator or None, optional, default: None
            An optional random number generator to use.
        """

        self.scales = scales
        if self.scales is None:
            self.scales = np.array([0.25, 0.5, 0.5, 0.5])
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.rng = rng
        if self.rng is None:
            self.rng = np.random.default_rng()

    def prior(self):
        """Generates a random draw from a 4-dimensional Gaussian prior distribution with a
        spherical convariance matrix. The parameters represent a robot's arm configuration,
        with the first parameter indicating the arm's height and the remaining three are
        angles.

        Returns
        -------
        params : np.ndarray of shape (4, )
            A single draw from the 4-dimensional Gaussian prior.
        """

        return self.rng.normal(loc=0, scale=self.scales)

    def observation_model(self, params: np.ndarray):
        """Returns the 2D coordinates of a robot arm given parameter vector.
        The first parameter represents the arm's height and the remaining three
        correspond to angles.

        Parameters
        ----------
        params   : np.ndarray of shape (params, )
            The four model parameters which will determine the coordinates

        Returns
        -------
        x : np.ndarray of shape (2, )
            The 2D coordinates of the arm
        """

        # Determine 2D position
        x1 = self.l1 * np.sin(params[1])
        x1 += self.l2 * np.sin(params[1] + params[2])
        x1 += self.l3 * np.sin(params[1] + params[2] + params[3]) + params[0]
        x2 = self.l1 * np.cos(params[1])
        x2 += self.l2 * np.cos(params[1] + params[2])
        x2 += self.l3 * np.cos(params[1] + params[2] + params[3])
        return np.array([x1, x2])
