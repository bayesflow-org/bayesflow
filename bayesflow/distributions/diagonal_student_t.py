import keras

import math
import numpy as np

from bayesflow.types import Shape, Tensor
from bayesflow.utils import expand_tile
from bayesflow.utils.decorators import allow_batch_size
from bayesflow.utils.serialization import serializable, serialize

from .distribution import Distribution


@serializable
class DiagonalStudentT(Distribution):
    """Implements a backend-agnostic diagonal Student-t distribution."""

    def __init__(
        self,
        df: int | float,
        loc: int | float | np.ndarray | Tensor = 0.0,
        scale: int | float | np.ndarray | Tensor = 1.0,
        use_learnable_parameters: bool = False,
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
        use_learnable_parameters : bool, optional
            Whether to treat `loc` and `scale` as learnable parameters. Default is False.
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

        self.dim = None
        self.log_normalization_constant = None

        self.use_learnable_parameters = use_learnable_parameters

        if seed_generator is None:
            seed_generator = keras.random.SeedGenerator()

        self.seed_generator = seed_generator

    def build(self, input_shape: Shape) -> None:
        self.dim = int(input_shape[-1])

        # convert to tensor and broadcast if necessary
        self.loc = keras.ops.broadcast_to(self.loc, (self.dim,))
        self.loc = keras.ops.cast(self.loc, "float32")

        self.scale = keras.ops.broadcast_to(self.scale, (self.dim,))
        self.scale = keras.ops.cast(self.scale, "float32")

        self.log_normalization_constant = (
            -0.5 * self.dim * math.log(self.df)
            - 0.5 * self.dim * math.log(math.pi)
            - math.lgamma(0.5 * self.df)
            + math.lgamma(0.5 * (self.df + self.dim))
            - keras.ops.sum(keras.ops.log(self.scale))
        )

        if self.use_learnable_parameters:
            self._loc = self.add_weight(
                shape=keras.ops.shape(self.loc),
                initializer=keras.initializers.get(self.loc),
                dtype="float32",
            )
            self._scale = self.add_weight(
                shape=keras.ops.shape(self.scale),
                initializer=keras.initializers.get(self.scale),
                dtype="float32",
            )
        else:
            self._loc = self.loc
            self._scale = self.scale

    def log_prob(self, samples: Tensor, *, normalize: bool = True) -> Tensor:
        mahalanobis_term = keras.ops.sum((samples - self._loc) ** 2 / self._scale**2, axis=-1)
        result = -0.5 * (self.df + self.dim) * keras.ops.log1p(mahalanobis_term / self.df)

        if normalize:
            result += self.log_normalization_constant

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

        return self._loc + self._scale * normal_samples * keras.ops.sqrt(self.df / chi2_samples)

    def get_config(self):
        base_config = super().get_config()

        config = {
            "df": self.df,
            "loc": self.loc,
            "scale": self.scale,
            "use_learnable_parameters": self.use_learnable_parameters,
            "seed_generator": self.seed_generator,
        }

        return base_config | serialize(config)
