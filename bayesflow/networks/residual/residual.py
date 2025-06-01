from collections.abc import Sequence

import keras

from bayesflow.types import Tensor
from bayesflow.utils import sequential_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ..sequential import Sequential


@serializable("bayesflow.networks")
class Residual(Sequential):
    def __init__(self, *layers: keras.Layer, **kwargs):
        if len(layers) == 1 and isinstance(layers[0], Sequence):
            layers = layers[0]
        super().__init__(list(layers), **sequential_kwargs(kwargs))
        self.projector = keras.layers.Dense(units=None, name="projector")

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = sequential_kwargs(base_config)

        config = {
            "layers": self.layers,
            "projector": self.projector,
        }

        return base_config | serialize(config)

    def build(self, input_shape=None):
        if self.built:
            # building when the network is already built can cause issues with serialization
            # see https://github.com/keras-team/keras/issues/21147
            return

        # we have to avoid calling super().build() because this causes
        # shape errors when building on non-sets but doing inference on sets
        # this is a work-around for https://github.com/keras-team/keras/issues/21158
        output_shape = input_shape
        for layer in self._layers:
            layer.build(output_shape)
            output_shape = layer.compute_output_shape(output_shape)

        self.projector.units = output_shape[-1]
        self.projector.build(input_shape)

    def call(self, x: Tensor, training: bool = None, mask: bool = None) -> Tensor:
        return self.projector(x) + super().call(x, training=training, mask=mask)
