from collections.abc import Sequence
import keras

from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.networks")
class Sequential(keras.Layer):
    """
    A custom sequential model for managing a sequence of Keras layers.

    This class extends `keras.Layer` and provides functionality for building,
    calling, and serializing a sequence of layers. Unlike `keras.Sequential`,
    this implementation allows for more flexibility in handling layer arguments
    and supports custom serialization through the `@serializable` decorator.

    Parameters
    ----------
    layers : keras.Layer or Sequence[keras.Layer]
        A single Keras layer or a sequence of Keras layers to be managed by this model.
    **kwargs : dict
        Additional keyword arguments passed to the base `keras.Layer` class.

    Notes
    -----
    - This class differs from `keras.Sequential` in that it does not assume a strict
      linear stack of layers and provides custom methods for serialization and
      configuration.
    - It is designed to integrate with the BayesFlow framework and supports
      additional utilities like `layer_kwargs`.
    """
    def __init__(self, *layers: keras.Layer | Sequence[keras.Layer], **kwargs):
        super().__init__(**layer_kwargs(kwargs))
        if len(layers) == 1 and isinstance(layers[0], Sequence):
            layers = layers[0]

        self._layers = layers

    def build(self, input_shape):
        if self.built:
            # building when the network is already built can cause issues with serialization
            # see https://github.com/keras-team/keras/issues/21147
            return

        for layer in self._layers:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)

    def call(self, inputs, training=None, mask=None):
        x = inputs
        for layer in self._layers:
            kwargs = self._make_kwargs_for_layer(layer, training, mask)
            x = layer(x, **kwargs)
        return x

    def compute_output_shape(self, input_shape):
        for layer in self._layers:
            input_shape = layer.compute_output_shape(input_shape)

        return input_shape

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "layers": [serialize(layer) for layer in self._layers],
        }

        return base_config | config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    @property
    def layers(self):
        return self._layers

    @staticmethod
    def _make_kwargs_for_layer(layer, training, mask):
        kwargs = {}
        if layer._call_has_mask_arg:
            kwargs["mask"] = mask
        if layer._call_has_training_arg and training is not None:
            kwargs["training"] = training
        return kwargs
