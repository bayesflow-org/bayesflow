from typing import Literal

import keras

from bayesflow.networks.unet.blocks import SimpleNorm
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.networks")
class DoubleConv(keras.Layer):
    """A double-convolution block with configurable normalization ordering.

    Uses pre-activation ordering (Norm -> Act -> Conv) by default, switching
    to post-activation ordering (Conv -> Act -> BN) when ``norm="batch"``.
    The last layer is zero-initialized so the block starts as near-identity
    when used inside a residual wrapper.
    """

    def __init__(
        self,
        width: int,
        norm: Literal["layer", "group", "batch"] | None = "group",
        groups: int | None = 8,
        dropout: float = None,
        activation: str = "mish",
        **kwargs,
    ):
        if norm == "batch":
            # Post-activation: Conv → Act → BN
            # https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
            layers = [
                keras.layers.Conv2D(width, 3, padding="same"),
                keras.layers.Activation(activation),
                SimpleNorm(method=norm),
                keras.layers.Conv2D(width, 3, padding="same"),
                keras.layers.Activation(activation),
                SimpleNorm(method=norm, gamma_initializer="zeros"),
                keras.layers.Dropout(0.0 if dropout is None else dropout),
            ]
        else:
            # Pre-activation: Norm → Act → Conv
            layers = [
                SimpleNorm(method=norm, groups=groups, center=True, scale=True),
                keras.layers.Activation(activation),
                keras.layers.Conv2D(width, 3, padding="same"),
                SimpleNorm(method=norm, groups=groups, center=True, scale=True),
                keras.layers.Activation(activation),
                keras.layers.Dropout(0.0 if dropout is None else dropout),
                keras.layers.Conv2D(width, 3, padding="same", kernel_initializer="zeros"),
            ]

        super().__init__(layers, **layer_kwargs(kwargs))

        self.width = width
        self.norm = norm
        self.groups = groups
        self.dropout = dropout
        self.activation = activation

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
        }

        return base_config | serialize(config)
