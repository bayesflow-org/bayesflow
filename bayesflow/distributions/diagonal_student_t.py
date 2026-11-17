import math

import numpy as np

import keras
from keras import ops

from bayesflow.types import Shape, Tensor
from bayesflow.utils import expand_tile
from bayesflow.utils.decorators import allow_batch_size
from bayesflow.utils.keras_utils import resolve_seed
from bayesflow.utils.serialization import serializable, serialize

from .distribution import Distribution


@serializable("bayesflow.distributions")
class DiagonalStudentT(Distribution):
    """Implements a backend-agnostic diagonal Student-t distribution."""

    def __init__(
        self,
        df: int | float = 25,
        loc: int | float | np.ndarray | Tensor = 0.0,
        scale: int | float | np.ndarray | Tensor = 1.0,
        trainable_parameters: bool = False,
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
        df : int or float, optional
            Degrees of freedom for the Student's t-distribution. Lower values result in
            heavier tails, making it more robust to outliers. Default is 25.
        loc : int, float, np.ndarray, or Tensor, optional
            The location parameter (mean) of the distribution. Default is 0.0.
        scale : int, float, np.ndarray, or Tensor, optional
            The scale parameter (standard deviation) of the distribution. Default is 1.0.
        trainable_parameters : bool, optional
            Whether to treat `loc` and `scale` as trainable parameters. Default is False.
        **kwargs
            Additional keyword arguments passed to the base `Distribution` class.
        """

        super().__init__(**kwargs)

        self.df = df
        self.loc = loc
        self.scale = scale

        self.trainable_parameters = trainable_parameters

        self.seed_generator = keras.random.SeedGenerator()

        self.dim = None
        self._loc = None
        self._scale = None

    def build(self, input_shape: Shape) -> None:
        if self.built:
            return

        self.dim = input_shape[-1]
        dtype = keras.config.floatx()

        # convert to tensor and broadcast if necessary
        self.loc = ops.cast(ops.broadcast_to(self.loc, (self.dim,)), dtype)
        self.scale = ops.cast(ops.broadcast_to(self.scale, (self.dim,)), dtype)

        if self.trainable_parameters:
            self._loc = self.add_weight(
                shape=ops.shape(self.loc),
                initializer=keras.initializers.get(keras.ops.copy(self.loc)),
                dtype=dtype,
                trainable=True,
            )
            self._scale = self.add_weight(
                shape=ops.shape(self.scale),
                initializer=keras.initializers.get(keras.ops.copy(self.scale)),
                dtype=dtype,
                trainable=True,
            )
        else:
            self._loc = self.loc
            self._scale = self.scale

    def log_prob(
        self,
        samples: Tensor,
        *,
        normalize: bool = True,
        mask: Tensor | None = None,
    ) -> Tensor:
        if mask is None:
            event_dim = self.dim
            mahalanobis_term = ops.sum((samples - self._loc) ** 2 / self._scale**2, axis=-1)
            log_scale = ops.sum(ops.log(self._scale))
        else:
            mask = ops.cast(ops.broadcast_to(mask, ops.shape(samples)), "bool")
            event_dim = ops.sum(ops.cast(mask, ops.dtype(samples)), axis=-1)
            squared_distance = (samples - self._loc) ** 2 / self._scale**2
            mahalanobis_term = ops.sum(
                ops.where(mask, squared_distance, ops.zeros_like(squared_distance)),
                axis=-1,
            )
            log_scale_terms = ops.broadcast_to(ops.log(self._scale), ops.shape(samples))
            log_scale = ops.sum(
                ops.where(mask, log_scale_terms, ops.zeros_like(log_scale_terms)),
                axis=-1,
            )

        result = -0.5 * (self.df + event_dim) * ops.log1p(mahalanobis_term / self.df)

        if normalize:
            if mask is None:
                log_gamma_ratio = math.lgamma(0.5 * (self.df + self.dim)) - math.lgamma(0.5 * self.df)
            else:
                # Keras has no cross-backend lgamma op. The active dimension is
                # an integer in [0, self.dim], so select from exact constants.
                gamma_ratios = ops.convert_to_tensor(
                    [math.lgamma(0.5 * (self.df + dim)) - math.lgamma(0.5 * self.df) for dim in range(self.dim + 1)],
                    dtype=ops.dtype(samples),
                )
                log_gamma_ratio = ops.take(gamma_ratios, ops.cast(event_dim, "int32"))

            log_normalization_constant = (
                -0.5 * event_dim * math.log(self.df) - 0.5 * event_dim * math.log(math.pi) + log_gamma_ratio - log_scale
            )
            result += log_normalization_constant

        return result

    @allow_batch_size
    def sample(self, batch_shape: Shape, seed: int | keras.random.SeedGenerator | None = None) -> Tensor:
        seed = resolve_seed(seed, self.seed_generator)
        # As of writing this code, keras does not support the chi-square distribution
        # nor does it support a scale or rate parameter in Gamma. Hence, we use the relation:
        # chi-square(df) = Gamma(shape = 0.5 * df, scale = 2) = Gamma(shape = 0.5 * df, scale = 1) * 2
        chi2_samples = keras.random.gamma(batch_shape, alpha=0.5 * self.df, seed=seed) * 2.0

        # The chi-quare samples need to be repeated across self.dim
        # since for each element of batch_shape only one sample is created.
        chi2_samples = expand_tile(chi2_samples, n=self.dim, axis=-1)
        chi2_samples = keras.ops.reshape(chi2_samples, batch_shape + (self.dim,))

        normal_samples = keras.random.normal(batch_shape + (self.dim,), seed=seed)

        return self._loc + self._scale * normal_samples * ops.sqrt(self.df / chi2_samples)

    def get_config(self):
        base_config = super().get_config()

        config = {
            "df": self.df,
            "loc": self.loc,
            "scale": self.scale,
            "trainable_parameters": self.trainable_parameters,
        }

        return base_config | serialize(config)
