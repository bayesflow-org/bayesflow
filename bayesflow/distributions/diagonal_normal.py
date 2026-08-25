import math

import numpy as np

import keras
from keras import ops

from bayesflow.types import Shape, Tensor
from bayesflow.utils.decorators import allow_batch_size
from bayesflow.utils.keras_utils import resolve_seed
from bayesflow.utils.serialization import serializable, serialize

from .distribution import Distribution


@serializable("bayesflow.distributions")
class DiagonalNormal(Distribution):
    """Implements a backend-agnostic diagonal Gaussian distribution."""

    def __init__(
        self,
        mean: int | float | np.ndarray | Tensor = 0.0,
        std: int | float | np.ndarray | Tensor = 1.0,
        trainable_parameters: bool = False,
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
        trainable_parameters : bool, optional
            Whether to treat the mean and standard deviation as learnable parameters. Default is False.
        **kwargs
            Additional keyword arguments passed to the base `Distribution` class.

        """

        super().__init__(**kwargs)
        self.mean = mean
        self.std = std

        self.trainable_parameters = trainable_parameters
        self.seed_generator = keras.random.SeedGenerator()

        self.dim = None
        self._mean = None
        self._std = None

    def build(self, input_shape: Shape) -> None:
        if self.built:
            return

        self.dim = input_shape[-1]
        dtype = keras.config.floatx()

        self.mean = ops.cast(ops.broadcast_to(self.mean, (self.dim,)), dtype)
        self.std = ops.cast(ops.broadcast_to(self.std, (self.dim,)), dtype)

        if self.trainable_parameters:
            self._mean = self.add_weight(
                shape=ops.shape(self.mean),
                initializer=keras.initializers.get(keras.ops.copy(self.mean)),
                dtype=dtype,
                trainable=True,
            )
            self._std = self.add_weight(
                shape=ops.shape(self.std),
                initializer=keras.initializers.get(keras.ops.copy(self.std)),
                dtype=dtype,
                trainable=True,
            )
        else:
            self._mean = self.mean
            self._std = self.std

    def log_prob(self, samples: Tensor, *, normalize: bool = True) -> Tensor:
        result = -0.5 * ops.sum((samples - self._mean) ** 2 / self._std**2, axis=-1)

        if normalize:
            log_normalization_constant = -0.5 * self.dim * math.log(2.0 * math.pi) - ops.sum(ops.log(self._std))
            result += log_normalization_constant

        return result

    @allow_batch_size
    def sample(self, batch_shape: Shape, seed: int | keras.random.SeedGenerator | None = None) -> Tensor:
        seed = resolve_seed(seed, self.seed_generator)
        z = keras.random.normal(shape=batch_shape + (self.dim,), seed=seed)
        z = self._mean + self._std * z
        return z

    def get_config(self):
        base_config = super().get_config()

        config = {
            "mean": self.mean,
            "std": self.std,
            "trainable_parameters": self.trainable_parameters,
        }

        return base_config | serialize(config)
