import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import keras_kwargs


class Distribution(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**keras_kwargs(kwargs))

    def call(self, samples: Tensor) -> Tensor:
        return keras.ops.exp(self.log_prob(samples))

    def log_prob(self, samples: Tensor, *, normalize: bool = True) -> Tensor:
        raise NotImplementedError

    def sample(self, batch_shape: Shape) -> Tensor:
        raise NotImplementedError
