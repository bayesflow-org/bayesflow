from collections.abc import Callable
from typing import Literal

import keras

from bayesflow.networks.unet.blocks import SimpleNorm
from bayesflow.utils import model_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class DoubleConv(keras.Sequential):
    def __init__(
        self,
        width: int,
        norm: Literal["layer", "group", "batch"] | None = "group",
        groups: int | None = 8,
        dropout: float = None,
        activation: str = "swish",
        residual: bool = True,
        **kwargs,
    ):

        if norm == "batch":
            last_norm_kwargs = {"gamma_initializer": "zeros"} if residual else {}
            # Post-activation: Conv → Act → BN
            # https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
            layers = [
                keras.layers.Conv2D(width, 3, padding="same"),
                keras.layers.Activation(activation),
                SimpleNorm(method=norm),
                keras.layers.Conv2D(width, 3, padding="same"),
                keras.layers.Activation(activation),
                SimpleNorm(method=norm, **last_norm_kwargs),
                keras.layers.Dropout(0.0 if dropout is None else dropout),
            ]
        else:
            last_conv_init = "zeros" if residual else "glorot_uniform"
            # Pre-activation: Norm → Act → Conv
            layers = [
                SimpleNorm(method=norm, groups=groups, center=True, scale=True),
                keras.layers.Activation(activation),
                keras.layers.Conv2D(width, 3, padding="same"),
                SimpleNorm(method=norm, groups=groups, center=True, scale=True),
                keras.layers.Activation(activation),
                keras.layers.Dropout(0.0 if dropout is None else dropout),
                keras.layers.Conv2D(width, 3, padding="same", kernel_initializer=last_conv_init),
            ]

        super().__init__(layers, **model_kwargs(kwargs))

        self.width = width
        self.norm = norm
        self.groups = groups
        self.dropout = dropout
        self.activation = activation
        self.residual = residual

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()

        config = {
            "width": self.width,
            "norm": self.norm,
            "dropout": self.dropout,
            "activation": self.activation,
            "residual": self.residual,
        }

        return base_config | serialize(config)

    def build(self, input_shape=None):
        # set the padding so the output is max-poolable
        *batch_shape, height, width, channels = input_shape

        padding = [height % 2, width % 2]
        self._layers.insert(0, keras.layers.ZeroPadding2D(padding=padding))

        super().build(input_shape)
