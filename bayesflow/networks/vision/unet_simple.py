from typing import Literal, Callable, Sequence

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, deserialize, serialize


@serializable("bayesflow.networks")
class UNet(keras.Layer):
    def __init__(
        self,
        *,
        **kwargs,
    ):
        super().__init__(**kwargs)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = super().get_config()
        base = layer_kwargs(base)
        cfg = {
        }
        return base | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return
        # some build logic here
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        # some compute output shape logic here
        return input_shape

    def call(
        self,
        inputs: tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, None],
        training: bool=None,
        mask=None
    ) -> Tensor:
        x, t, conditions = inputs
        # some call logic here
        return x
