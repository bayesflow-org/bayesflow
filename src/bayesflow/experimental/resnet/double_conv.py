from collections.abc import Callable
import keras

from bayesflow.utils import model_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class DoubleConv(keras.Sequential):
    def __init__(
        self,
        width: int,
        use_batchnorm: bool = True,
        dropout: float = None,
        activation: str | Callable[[], keras.Layer] = "relu",
        **kwargs,
    ):
        layers = []

        layers.append(keras.layers.Conv2D(width, 3, padding="same"))

        activation = keras.activations.get(activation)
        if not isinstance(activation, keras.Layer):
            activation = keras.layers.Activation(activation)

        layers.append(activation)

        if use_batchnorm:
            # we apply this after the activation due to
            # https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
            layers.append(keras.layers.BatchNormalization())

        if dropout is not None and dropout > 0:
            layers.append(keras.layers.Dropout(dropout))

        layers.append(keras.layers.Conv2D(width, 3, padding="same"))

        if use_batchnorm:
            layers.append(keras.layers.BatchNormalization())

        super().__init__(layers, **model_kwargs(kwargs))

        self.width = width
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.activation = activation

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()

        config = {
            "width": self.width,
            "use_batchnorm": self.use_batchnorm,
            "dropout": self.dropout,
            "activation": self.activation,
        }

        return base_config | serialize(config)

    def build(self, input_shape=None):
        # set the padding so the output is max-poolable
        *batch_shape, height, width, channels = input_shape

        padding = [height % 2, width % 2]
        self._layers.insert(0, keras.layers.ZeroPadding2D(padding=padding))

        super().build(input_shape)
