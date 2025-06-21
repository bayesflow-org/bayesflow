import math

import numpy as np

import keras
from keras import ops

from bayesflow.types import Shape, Tensor
from bayesflow.utils import expand_tile
from bayesflow.utils.decorators import allow_batch_size
from bayesflow.utils.serialization import serializable, serialize

from .distribution import Distribution


@serializable("bayesflow.distributions")
class DiagonalStudentT(Distribution):
    """Implements a backend-agnostic diagonal Student-t distribution."""

    def __init__(
        self,
        df: int | float,
        loc: int | float | np.ndarray | Tensor = 0.0,
        scale: int | float | np.ndarray | Tensor = 1.0,
        trainable_parameters: bool = False,
        seed_generator: keras.random.SeedGenerator = None,
        **kwargs,
    ):
        """
        Initializes a backend-agnostic Student's t-distribution with optional learnable parameters.

        This class represents a Student's t-distribution, which is useful for modeling heavy-tailed data.
        The distribution is parameterized by degrees of freedom (`df`), location (`loc`), and scale (`scale`).
        These parameters can either be fixed or learned during training.

        The class also supports random number generation with an optional seed for reproducibility.

        Parameters
        ----------
        df : int or float
            Degrees of freedom for the Student's t-distribution. Lower values result in
            heavier tails, making it more robust to outliers.
        loc : int, float, np.ndarray, or Tensor, optional
            The location parameter (mean) of the distribution. Default is 0.0.
        scale : int, float, np.ndarray, or Tensor, optional
            The scale parameter (standard deviation) of the distribution. Default is 1.0.
        trainable_parameters : bool, optional
            Whether to treat `loc` and `scale` as trainable parameters. Default is False.
        seed_generator : keras.random.SeedGenerator, optional
            A Keras seed generator for reproducible random sampling. If None, a new seed
            generator is created. Default is None.
        **kwargs
            Additional keyword arguments passed to the base `Distribution` class.
        """

        super().__init__(**kwargs)

        self.df = df
        self.loc = loc
        self.scale = scale

        self.trainable_parameters = trainable_parameters

        self.seed_generator = seed_generator or keras.random.SeedGenerator()

        self.dim = None
        self._loc = None
        self._scale = None

    def build(self, input_shape: Shape) -> None:
        if self.built:
            return

        self.dim = int(input_shape[-1])

        # convert to tensor and broadcast if necessary
        self.loc = ops.cast(ops.broadcast_to(self.loc, (self.dim,)), "float32")
        self.scale = ops.cast(ops.broadcast_to(self.scale, (self.dim,)), "float32")

        if self.trainable_parameters:
            self._loc = self.add_weight(
                shape=ops.shape(self.loc),
                initializer=keras.initializers.get(keras.ops.copy(self.loc)),
                dtype="float32",
                trainable=True,
            )
            self._scale = self.add_weight(
                shape=ops.shape(self.scale),
                initializer=keras.initializers.get(keras.ops.copy(self.scale)),
                dtype="float32",
                trainable=True,
            )
        else:
            self._loc = self.loc
            self._scale = self.scale

    def log_prob(self, samples: Tensor, *, normalize: bool = True) -> Tensor:
        mahalanobis_term = ops.sum((samples - self._loc) ** 2 / self._scale**2, axis=-1)
        result = -0.5 * (self.df + self.dim) * ops.log1p(mahalanobis_term / self.df)

        if normalize:
            log_normalization_constant = (
                -0.5 * self.dim * math.log(self.df)
                - 0.5 * self.dim * math.log(math.pi)
                - math.lgamma(0.5 * self.df)
                + math.lgamma(0.5 * (self.df + self.dim))
                - ops.sum(keras.ops.log(self._scale))
            )
            result += log_normalization_constant

        return result

    @allow_batch_size
    def sample(self, batch_shape: Shape) -> Tensor:
        # As of writing this code, keras does not support the chi-square distribution
        # nor does it support a scale or rate parameter in Gamma. Hence, we use the relation:
        # chi-square(df) = Gamma(shape = 0.5 * df, scale = 2) = Gamma(shape = 0.5 * df, scale = 1) * 2
        chi2_samples = keras.random.gamma(batch_shape, alpha=0.5 * self.df, seed=self.seed_generator) * 2.0

        # The chi-quare samples need to be repeated across self.dim
        # since for each element of batch_shape only one sample is created.
        chi2_samples = expand_tile(chi2_samples, n=self.dim, axis=-1)

        normal_samples = keras.random.normal(batch_shape + (self.dim,), seed=self.seed_generator)

        return self._loc + self._scale * normal_samples * ops.sqrt(self.df / chi2_samples)

    def get_config(self):
        base_config = super().get_config()

        config = {
            "df": self.df,
            "loc": self.loc,
            "scale": self.scale,
            "trainable_parameters": self.trainable_parameters,
            "seed_generator": self.seed_generator,
        }

        return base_config | serialize(config)
