import keras

from bayesflow.utils import layer_kwargs

from bayesflow.utils.serialization import deserialize


class InvertibleLayer(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**layer_kwargs(kwargs))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def call(self, *args, **kwargs):
        # we cannot provide a default implementation for this
        #  because the signature of layer.call() is used to
        #  determine the arguments to layer.build()
        raise NotImplementedError

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def _inverse(self, *args, **kwargs):
        raise NotImplementedError
