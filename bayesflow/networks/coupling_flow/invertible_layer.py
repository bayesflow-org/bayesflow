import keras

from bayesflow.utils import keras_kwargs


class InvertibleLayer(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**keras_kwargs(kwargs))

    def call(self, *args, **kwargs):
        # we cannot provide a default implementation for this
        #  because the signature of layer.call() is used to
        #  determine the arguments to layer.build()
        raise NotImplementedError

    def _forward(self, *args, **kwargs):
        raise NotImplementedError

    def _inverse(self, *args, **kwargs):
        raise NotImplementedError
