import math

import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Shape, Tensor
from bayesflow.links import PositiveSemiDefinite
from bayesflow.utils import logging

from .parametric_distribution_score import ParametricDistributionScore


@serializable(package="bayesflow.scores")
class MultivariateNormalScore(ParametricDistributionScore):
    r""":math:`S(\hat p_{\mu, \Sigma}, \theta; k) = \log( \mathcal N (\theta; \mu, \Sigma))`

    Scores a predicted mean and covariance matrix with the log-score of the probability of the materialized value.
    """

    def __init__(self, dim: int = None, links: dict = None, **kwargs):
        super().__init__(links=links, **kwargs)

        self.dim = dim
        self.links = links or {"covariance": PositiveSemiDefinite()}
        self.config = {"dim": dim}

        logging.warning("MultivariateNormalScore is unstable.")

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config

    def get_head_shapes_from_target_shape(self, target_shape: Shape) -> dict[str, Shape]:
        self.dim = target_shape[-1]
        return dict(mean=(self.dim,), covariance=(self.dim, self.dim))

    def log_prob(self, x: Tensor, mean: Tensor, covariance: Tensor) -> Tensor:
        """
        Compute the log probability density of a multivariate Gaussian distribution.

        This function calculates the log probability density for each sample in `x` under a
        multivariate Gaussian distribution with the given `mean` and `covariance`.

        The computation includes the determinant of the covariance matrix, its inverse, and the quadratic
        form in the exponential term of the Gaussian density function.

        Parameters
        ----------
        x : Tensor
            A tensor of input samples for which the log probability density is computed.
            The shape should be compatible with broadcasting against `mean`.
        mean : Tensor
            A tensor representing the mean of the multivariate Gaussian distribution.
        covariance : Tensor
            A tensor representing the covariance matrix of the multivariate Gaussian distribution.

        Returns
        -------
        Tensor
            A tensor containing the log probability densities for each sample in `x` under the
            given Gaussian distribution.
        """
        diff = x[:, None, :] - mean
        inv_covariance = keras.ops.inv(covariance)
        log_det_covariance = keras.ops.slogdet(covariance)[1]  # Only take the log of the determinant part

        # Compute the quadratic term in the exponential of the multivariate Gaussian
        quadratic_term = keras.ops.einsum("...i,...ij,...j->...", diff, inv_covariance, diff)

        # Compute the log probability density
        log_prob = -0.5 * (self.dim * keras.ops.log(2 * math.pi) + log_det_covariance + quadratic_term)

        return log_prob

    def sample(self, batch_shape: Shape, mean: Tensor, covariance: Tensor) -> Tensor:
        """
        Generate samples from a multivariate Gaussian distribution.

        This function samples from a multivariate Gaussian distribution with the given `mean`
        and `covariance` using the Cholesky decomposition method. Independent standard normal
        samples are transformed using the Cholesky factor of the covariance matrix to generate
        correlated samples.

        Parameters
        ----------
        batch_shape : Shape
            A tuple specifying the batch size and the number of samples to generate.
        mean : Tensor
            A tensor representing the mean of the multivariate Gaussian distribution.
            Must have shape (batch_size, D), where D is the dimensionality of the distribution.
        covariance : Tensor
            A tensor representing the covariance matrix of the multivariate Gaussian distribution.
            Must have shape (batch_size, D, D), where D is the dimensionality.

        Returns
        -------
        Tensor
            A tensor of shape (batch_size, num_samples, D) containing the generated samples.
        """
        batch_size, num_samples = batch_shape
        dim = mean.shape[-1]
        assert mean.shape == (batch_size, dim), "mean must have shape (batch_size, D)"
        assert covariance.shape == (batch_size, dim, dim), "covariance must have shape (batch_size, D, D)"

        # Use Cholesky decomposition to generate samples
        cholesky_factor = keras.ops.cholesky(covariance)
        normal_samples = keras.random.normal((*batch_shape, dim))

        scaled_normal = keras.ops.einsum("ijk,ilk->ilj", cholesky_factor, normal_samples)
        samples = mean[:, None, :] + scaled_normal

        return samples
