import math

import keras

from bayesflow.types import Shape, Tensor
from bayesflow.links import PositiveDefinite
from bayesflow.utils.serialization import serializable

from .parametric_distribution_score import ParametricDistributionScore


@serializable("bayesflow.scores")
class MultivariateNormalScore(ParametricDistributionScore):
    r""":math:`S(\hat p_{\mu, \Sigma}, \theta; k) = -\log( \mathcal N (\theta; \mu, \Sigma))`

    Scores a predicted mean and covariance matrix with the log-score of the probability of the materialized value.
    """

    NOT_TRANSFORMING_LIKE_VECTOR_WARNING = ("covariance",)
    """
    Marks head for covariance matrix as an exception for adapter transformations.

    This variable contains names of prediction heads that should lead to a warning when the adapter is applied
    in inverse direction to them.

    For more information see :py:class:`ScoringRule`.
    """

    def __init__(self, dim: int = None, links: dict = None, **kwargs):
        super().__init__(links=links, **kwargs)

        self.dim = dim
        self.links = links or {"covariance": PositiveDefinite()}

        self.config = {"dim": dim}

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
        diff = x - mean
        precision = keras.ops.inv(covariance)
        log_det_covariance = keras.ops.slogdet(covariance)[1]  # Only take the log of the determinant part

        # Compute the quadratic term in the exponential of the multivariate Gaussian
        quadratic_term = keras.ops.einsum("...i,...ij,...j->...", diff, precision, diff)

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
        if len(batch_shape) == 1:
            batch_shape = (1,) + tuple(batch_shape)
        batch_size, num_samples = batch_shape
        dim = keras.ops.shape(mean)[-1]
        if keras.ops.shape(mean) != (batch_size, dim):
            raise ValueError(f"mean must have shape (batch_size, {dim}), but got {keras.ops.shape(mean)}")

        if keras.ops.shape(covariance) != (batch_size, dim, dim):
            raise ValueError(
                f"covariance must have shape (batch_size, {dim}, {dim}), but got {keras.ops.shape(covariance)}"
            )

        # Use Cholesky decomposition to generate samples
        cholesky_factor = keras.ops.cholesky(covariance)
        normal_samples = keras.random.normal((*batch_shape, dim))

        scaled_normal = keras.ops.einsum("ijk,ilk->ilj", cholesky_factor, normal_samples)
        samples = mean[:, None, :] + scaled_normal

        return samples
