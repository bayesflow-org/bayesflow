import keras

from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable, serialize


@serializable("bayesflow.networks")
class Downsample(keras.Layer):
    """Temporal downsampling for transformer sequence inputs."""

    def __init__(
        self,
        factor: int | None = None,
        filters: int | None = None,
        kernel_size: int = 3,
        activation: str = "gelu",
        kernel_initializer: str = "orthogonal",
        use_bias: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if isinstance(factor, bool) or (factor is not None and (not isinstance(factor, int) or factor < 1)):
            raise ValueError(f"factor must be None or a positive integer, got {factor!r}.")

        self.factor = None if factor in (None, 1) else factor
        self.filters = filters
        self.kernel_size = kernel_size
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.use_bias = use_bias

        if self.factor is None:
            self.feature_downsampler = keras.layers.Identity()
            self.mask_downsampler = None
            self.time_downsampler = None
        else:
            self.feature_downsampler = None
            self.mask_downsampler = keras.layers.MaxPooling1D(
                pool_size=self.factor,
                strides=self.factor,
                padding="same",
            )
            self.time_downsampler = keras.layers.AveragePooling1D(
                pool_size=self.factor,
                strides=self.factor,
                padding="same",
            )

    def build(self, input_shape):
        if self.built:
            return

        if self.filters is None:
            self.filters = input_shape[-1]

        if self.factor is not None and self.feature_downsampler is None:
            self.feature_downsampler = keras.layers.Conv1D(
                filters=self.filters,
                kernel_size=self.kernel_size,
                strides=self.factor,
                padding="same",
                activation=self.activation,
                kernel_initializer=self.kernel_initializer,
                use_bias=self.use_bias,
            )

        self.feature_downsampler.build(input_shape)

    def call(self, x: Tensor) -> Tensor:
        return self.feature_downsampler(x)

    def downsample_mask(self, mask: Tensor | None) -> Tensor | None:
        if mask is None or self.mask_downsampler is None:
            return mask

        mask = keras.ops.squeeze(mask, axis=-2)
        mask = keras.ops.cast(mask, dtype=keras.config.floatx())
        mask = keras.ops.expand_dims(mask, axis=-1)
        mask = self.mask_downsampler(mask)
        mask = keras.ops.squeeze(mask, axis=-1)
        return keras.ops.expand_dims(mask > 0.0, axis=1)

    def downsample_time(self, t: Tensor | None) -> Tensor | None:
        """Average-pool explicit time values independently of feature masks."""

        if t is None or self.time_downsampler is None:
            return t

        t = keras.ops.expand_dims(t, axis=-1)
        t = self.time_downsampler(t)
        return keras.ops.squeeze(t, axis=-1)

    def get_config(self) -> dict:
        base_config = super().get_config()
        return base_config | serialize(
            {
                "factor": self.factor,
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "activation": self.activation,
                "kernel_initializer": self.kernel_initializer,
                "use_bias": self.use_bias,
            }
        )
