import keras

from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize, deserialize


@serializable("bayesflow.networks")
class FiLM(keras.Layer):
    """Feature-wise Linear Modulation: y = (1 + gamma) * x + beta."""

    def __init__(self, units: int, *, kernel_initializer="he_normal", **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.kernel_initializer = kernel_initializer
        self.to_gamma_beta = keras.layers.Dense(
            2 * self.units, kernel_initializer=kernel_initializer, name="to_gamma_beta"
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = super().get_config()
        base = layer_kwargs(base)
        cfg = {"units": self.units, "kernel_initializer": self.kernel_initializer}
        return base | serialize(cfg)

    def call(self, x, t_emb):
        gb = self.to_gamma_beta(t_emb)
        gamma, beta = keras.ops.split(gb, 2, axis=-1)
        return (1.0 + gamma) * x + beta
