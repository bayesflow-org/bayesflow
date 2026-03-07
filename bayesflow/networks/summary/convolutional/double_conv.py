from typing import Literal

import keras

from bayesflow.networks.components.unet.blocks import SimpleNorm
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
        residual: bool = True,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        if norm == "batch":
            # Post-activation: Conv → Act → BN
            # https://github.com/keras-team/keras/issues/1802#issuecomment-187966878
            gamma_initializer = "zeros" if residual else "ones"
            conv_layers = [
                keras.layers.Conv2D(width, 3, padding="same"),
                keras.layers.Activation(activation),
                SimpleNorm(method=norm),
                keras.layers.Conv2D(width, 3, padding="same"),
                keras.layers.Activation(activation),
                SimpleNorm(method=norm, gamma_initializer=gamma_initializer),
                keras.layers.Dropout(0.0 if dropout is None else dropout),
            ]
        else:
            # Pre-activation: Norm → Act → Conv
            kernel_initializer = "zeros" if residual else "glorot_uniform"
            conv_layers = [
                SimpleNorm(method=norm, groups=groups, center=True, scale=True),
                keras.layers.Activation(activation),
                keras.layers.Conv2D(width, 3, padding="same"),
                SimpleNorm(method=norm, groups=groups, center=True, scale=True),
                keras.layers.Activation(activation),
                keras.layers.Dropout(0.0 if dropout is None else dropout),
                keras.layers.Conv2D(width, 3, padding="same", kernel_initializer=kernel_initializer),
            ]

        self.conv_layers = conv_layers
        self.width = width
        self.norm = norm
        self.groups = groups
        self.dropout = dropout
        self.activation = activation
        self.residual = residual
        self.proj = keras.layers.Conv2D(width, 1, padding="same") if residual else keras.layers.Identity()

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()

        config = {
            "width": self.width,
            "norm": self.norm,
            "groups": self.groups,
            "dropout": self.dropout,
            "activation": self.activation,
            "residual": self.residual,
        }

        return base_config | serialize(config)

    def build(self, input_shape):
        shape = input_shape
        self.proj.build(shape)
        shape = self.proj.compute_output_shape(shape)
        for layer in self.conv_layers:
            layer.build(shape)
            shape = layer.compute_output_shape(shape)

    def compute_output_shape(self, input_shape):
        shape = input_shape
        shape = self.proj.compute_output_shape(shape)
        for layer in self.conv_layers:
            shape = layer.compute_output_shape(shape)
        return shape

    def call(self, x, training=False, **kwargs):
        x = self.proj(x)
        h = x
        for layer in self.conv_layers:
            h = layer(h, training=training)
        if self.residual:
            return x + h
        return h
