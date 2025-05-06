import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, deserialize


@serializable("bayesflow.distributions")
class Distribution(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**layer_kwargs(kwargs))

    def call(self, samples: Tensor) -> Tensor:
        return keras.ops.exp(self.log_prob(samples))

    def log_prob(self, samples: Tensor, *, normalize: bool = True) -> Tensor:
        raise NotImplementedError

    def sample(self, batch_shape: Shape) -> Tensor:
        raise NotImplementedError

    def compute_output_shape(self, input_shape: Shape) -> Shape:
        return keras.ops.shape(self.sample(input_shape[0:1]))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
