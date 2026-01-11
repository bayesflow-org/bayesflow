import keras

from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize, deserialize


@serializable("bayesflow.networks")
class FiLM(keras.Layer):
    """Feature-wise Linear Modulation: y = (1 + gamma) * x + beta."""

    def __init__(self, units: int, *, kernel_initializer="he_normal", use_gamma: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.kernel_initializer = kernel_initializer
        self.use_gamma = bool(use_gamma)

        if self.use_gamma:
            units *= 2  # for both gamma and beta
        self.to_gamma_beta = keras.layers.Dense(units, kernel_initializer=kernel_initializer, name="to_gamma_beta")

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = super().get_config()
        base = layer_kwargs(base)
        config = {
            "units": self.units,
            "kernel_initializer": self.kernel_initializer,
            "use_gamma": self.use_gamma,
        }
        return base | serialize(config)

    def build(self, input_shape):
        if self.built:
            return
        x_shape, t_emb_shape = input_shape

        # Verify x has correct feature dimension
        if x_shape[-1] != self.units:
            raise ValueError(f"FiLM layer expects input with {self.units} features, but got {x_shape[-1]}")

        # Build the projection from time embedding to gamma/beta
        self.to_gamma_beta.build(t_emb_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, _ = input_shape
        return x_shape

    def call(self, inputs, training=None, mask=None):
        x, t_emb = inputs
        gb = self.to_gamma_beta(t_emb)
        if self.use_gamma:
            gamma, beta = keras.ops.split(gb, 2, axis=-1)
            return (1.0 + gamma) * x + beta
        else:
            return x + gb
