import keras
from keras.saving import register_keras_serializable as serializable

import math
import numpy as np

from bayesflow.types import Shape, Tensor
from bayesflow.utils.decorators import allow_batch_size

from .distribution import Distribution


@serializable(package="bayesflow.distributions")
class DiagonalNormal(Distribution):
    """Implements a backend-agnostic diagonal Gaussian distribution."""

    def __init__(
        self,
        mean: int | float | np.ndarray | Tensor = 0.0,
        std: int | float | np.ndarray | Tensor = 1.0,
        use_learnable_parameters: bool = False,
        seed_generator: keras.random.SeedGenerator = None,
        **kwargs,
    ):
        """
        Initializes a backend-agnostic diagonal Gaussian distribution with optional learnable parameters.

        This class represents a Gaussian distribution with a diagonal covariance matrix, allowing for efficient
        sampling and density evaluation.

        The mean and standard deviation can be specified as fixed values or learned during training. The class also
        supports random number generation with an optional seed for reproducibility.

        Parameters
        ----------
        mean : int, float, np.ndarray, or Tensor, optional
            The mean of the Gaussian distribution. Can be a scalar or a tensor. Default is 0.0.
        std : int, float, np.ndarray, or Tensor, optional
            The standard deviation of the Gaussian distribution. Can be a scalar or a tensor.
            Default is 1.0.
        use_learnable_parameters : bool, optional
            Whether to treat the mean and standard deviation as learnable parameters. Default is False.
        seed_generator : keras.random.SeedGenerator, optional
            A Keras seed generator for reproducible random sampling. If None, a new seed
            generator is created. Default is None.
        **kwargs
            Additional keyword arguments passed to the base `Distribution` class.

        """

        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

        self.dim = None
        self.log_normalization_constant = None

        self.use_learnable_parameters = use_learnable_parameters

        if seed_generator is None:
            seed_generator = keras.random.SeedGenerator()

        self.seed_generator = seed_generator

    def build(self, input_shape: Shape) -> None:
        self.dim = int(input_shape[-1])

        # convert to tensor and broadcast if necessary
        self.mean = keras.ops.broadcast_to(self.mean, (self.dim,))
        self.mean = keras.ops.cast(self.mean, "float32")

        self.std = keras.ops.broadcast_to(self.std, (self.dim,))
        self.std = keras.ops.cast(self.std, "float32")

        self.log_normalization_constant = -0.5 * self.dim * math.log(2.0 * math.pi) - keras.ops.sum(
            keras.ops.log(self.std)
        )

        if self.use_learnable_parameters:
            mean = self.mean
            self.mean = self.add_weight(
                shape=keras.ops.shape(mean),
                initializer="zeros",
                dtype="float32",
            )
            self.mean.assign(mean)

            std = self.std
            self.std = self.add_weight(
                shape=keras.ops.shape(std),
                initializer="ones",
                dtype="float32",
            )
            self.std.assign(std)

    def log_prob(self, samples: Tensor, *, normalize: bool = True) -> Tensor:
        result = -0.5 * keras.ops.sum((samples - self.mean) ** 2 / self.std**2, axis=-1)

        if normalize:
            result += self.log_normalization_constant

        return result

    @allow_batch_size
    def sample(self, batch_shape: Shape) -> Tensor:
        return self.mean + self.std * keras.random.normal(shape=batch_shape + (self.dim,), seed=self.seed_generator)
