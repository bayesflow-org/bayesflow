from collections.abc import Callable, Sequence
import keras

from bayesflow.utils import model_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from bayesflow.networks.residual import Residual
from .double_conv import DoubleConv


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class ResNet(keras.Sequential):
    """
    Implements the ResNet architecture.

    Note that we still apply dropout and activation to the output and do not flatten it,
    so you will need to flatten it yourself and apply at least one linear layer after this network.
    """

    def __init__(
        self,
        widths: Sequence[int],
        use_batchnorm: bool = True,
        dropout: float = 0.05,
        activation: str | Callable[[], keras.Layer] = "relu",
        **kwargs,
    ):
        layers = []

        for width in widths:
            layer = DoubleConv(width, use_batchnorm=use_batchnorm, dropout=dropout, activation=activation)
            layer = Residual(layer)
            act = keras.activations.get(activation)
            if not isinstance(act, keras.Layer):
                act = keras.layers.Activation(act)
            maxpool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))

            layers.append(layer)
            layers.append(act)
            layers.append(maxpool)

        super().__init__(layers, **model_kwargs(kwargs))

        self.widths = widths
        self.use_batchnorm = use_batchnorm
        self.dropout = dropout
        self.activation = activation

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()

        config = {
            "widths": self.widths,
            "use_batchnorm": self.use_batchnorm,
            "dropout": self.dropout,
            "activation": self.activation,
        }

        return base_config | serialize(config)
