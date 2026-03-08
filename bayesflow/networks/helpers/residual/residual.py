from collections.abc import Sequence

import keras

from bayesflow.types import Tensor
from bayesflow.utils import sequential_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ..sequential import Sequential


@serializable("bayesflow.networks")
class Residual(Sequential):
    def __init__(self, *layers: keras.Layer, **kwargs):
        if len(layers) == 0 and "layers" in kwargs:
            # extract layers from kwargs, in case they were passed as a keyword argument
            layers = kwargs.pop("layers")
        elif len(layers) > 0 and "layers" in kwargs:
            raise ValueError("Layers passed both as positional argument and as keyword argument")
        elif len(layers) == 1 and isinstance(layers[0], Sequence):
            layers = layers[0]
        super().__init__(list(layers), **sequential_kwargs(kwargs))
        self.projector = None

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
            if layer.built:
                continue
            layer.build(output_shape)
            output_shape = layer.compute_output_shape(output_shape)

        self.projector = keras.layers.Dense(units=output_shape[-1], name="projector")
        self.projector.build(input_shape)

    def call(self, x: Tensor, training: bool = None, mask: bool = None) -> Tensor:
        return self.projector(x) + super().call(x, training=training, mask=mask)
