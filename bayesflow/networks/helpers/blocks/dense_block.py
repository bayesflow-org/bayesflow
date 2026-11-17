from typing import Literal, Callable

import keras

from bayesflow.utils import layer_kwargs
from bayesflow.types import Tensor
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.networks")
class DenseBlock(keras.Layer):
    """Single fully-connected hidden layer with optional residual skip,
    dropout, and normalization.

    Computes::

        h = dense(x)
        h = activation(h)
        h = dropout(h)  # if dropout > 0
        h = norm(h)  # if norm is not None
        h = projector(x) + h  # if residual

    Parameters
    ----------
    width : int
        Number of output units.
    activation : str or callable, optional
        Activation function. Default is ``"mish"``.
    kernel_initializer : str or keras.Initializer, optional
        Weight initializer for the Dense layer. Default is ``"he_normal"``.
    residual : bool, optional
        Add a skip connection from input to output. A learned projection
        is used when dimensions differ. Default is ``True``.
    dropout : float or None, optional
        Dropout rate. ``None`` or ``0`` disables dropout. Default is ``0.05``.
    norm : ``"batch"``, ``"layer"``, keras.Layer, or None, optional
        Normalization applied after activation (and after the residual
        addition, if enabled). Default is ``None``.
    **kwargs
        Extra keyword arguments forwarded to ``keras.Layer``.
    """

    def __init__(
        self,
        width: int,
        *,
        activation: str | Callable[[], keras.Layer] = "mish",
        kernel_initializer: str | keras.Initializer = "he_normal",
        residual: bool = True,
        dropout: Literal[0, None] | float = 0.05,
        norm: Literal["batch", "layer", "rms"] | keras.Layer = None,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        self.width = width
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.residual = residual
        self.dropout = dropout
        self.norm = norm

        # Internal dense layer
        self.dense = keras.layers.Dense(self.width, kernel_initializer=kernel_initializer, name="dense")

        # Optional dropout layer
        self.dropout_layer = None
        if dropout is not None and dropout > 0:
            self.dropout_layer = keras.layers.Dropout(dropout, name="dropout")

        # Non-linear activation
        activation = keras.activations.get(activation)
        if not isinstance(activation, keras.Layer):
            activation = keras.layers.Activation(activation)
        self.activation = activation

        # Optional normalization
        if norm == "batch":
            self.norm_layer = keras.layers.BatchNormalization(name="norm")
        elif norm == "layer":
            self.norm_layer = keras.layers.LayerNormalization(name="norm")
        elif norm == "rms":
            self.norm_layer = keras.layers.RMSNormalization(name="norm")
        elif isinstance(norm, str):
            raise ValueError(f"Unknown normalization strategy: {norm!r}.")
        elif isinstance(norm, keras.Layer):
            self.norm_layer = norm
        elif norm is None:
            self.norm_layer = None
        else:
            raise TypeError(f"Cannot infer norm from {norm!r} of type {type(norm)}.")

        # Residual projector (created in build if dims differ)
        self.projector = None

    def call(self, x: Tensor, training: bool = False):
        h = self.dense(x, training=training)
        h = self.activation(h)

        if self.dropout_layer is not None:
            h = self.dropout_layer(h, training=training)

        if self.residual:
            skip = x if self.projector is None else self.projector(x)
            h = skip + h

        if self.norm_layer is not None:
            h = self.norm_layer(h, training=training)

        return h

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = super().get_config()
        base = layer_kwargs(base)

        config = {
            "width": self.width,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "residual": self.residual,
            "dropout": self.dropout,
            "norm": self.norm,
        }
        return base | serialize(config)

    def build(self, input_shape):
        if self.built:
            return

        self.dense.build(input_shape)
        h_shape = self.dense.compute_output_shape(input_shape)
        self.activation.build(h_shape)

        if self.dropout_layer is not None:
            self.dropout_layer.build(h_shape)

        if self.norm_layer is not None:
            self.norm_layer.build(h_shape)

        if self.residual and input_shape[-1] != h_shape[-1]:
            self.projector = keras.layers.Dense(
                h_shape[-1], kernel_initializer=self.kernel_initializer, name="projector"
            )
            self.projector.build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.dense.compute_output_shape(input_shape)
