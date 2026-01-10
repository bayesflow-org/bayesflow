from typing import Literal, Callable

import keras

from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize
from ..embeddings import FiLM


@serializable("bayesflow.networks")
class ConditionalResidual(keras.Layer):
    """
    A single hidden block with optional residual connection and FiLM injection for conditional embedding.
    """

    def __init__(
        self,
        width: int,
        *,
        activation: str | Callable[[], keras.Layer] = "mish",
        kernel_initializer: str | keras.Initializer = "he_normal",
        residual: bool = True,
        dropout: Literal[0, None] | float = 0.05,
        norm: Literal["batch", "layer"] | keras.Layer = None,
        spectral_normalization: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.width = int(width)
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.residual = bool(residual)
        self.dropout = dropout
        self.norm = norm
        self.spectral_normalization = bool(spectral_normalization)

        dense = keras.layers.Dense(self.width, kernel_initializer=kernel_initializer, name="dense")
        if spectral_normalization:
            dense = keras.layers.SpectralNormalization(dense)
        self.dense = dense

        self.dropout_layer = None
        if dropout is not None and dropout > 0:
            self.dropout_layer = keras.layers.Dropout(dropout, name="dropout")

        act = keras.activations.get(activation)
        if not isinstance(act, keras.Layer):
            act = keras.layers.Activation(act)
        self.act = act

        if norm == "batch":
            self.norm_layer = keras.layers.BatchNormalization(name="norm")
        elif norm == "layer":
            self.norm_layer = keras.layers.LayerNormalization(name="norm")
        elif isinstance(norm, str):
            raise ValueError(f"Unknown normalization strategy: {norm!r}.")
        elif isinstance(norm, keras.Layer):
            self.norm_layer = norm
        elif norm is None:
            self.norm_layer = None
        else:
            raise TypeError(f"Cannot infer norm from {norm!r} of type {type(norm)}.")

        self.film = FiLM(self.width, kernel_initializer=kernel_initializer, name="film")
        self.projector = keras.layers.Dense(units=None, kernel_initializer=kernel_initializer, name="projector")

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = super().get_config()
        base = layer_kwargs(base)
        cfg = {
            "width": self.width,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "residual": self.residual,
            "dropout": self.dropout,
            "norm": self.norm,
            "spectral_normalization": self.spectral_normalization,
        }
        return base | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return

        x_shape, cond_shape = input_shape
        self.dense.build(x_shape)
        h_shape = self.dense.compute_output_shape(x_shape)

        if self.dropout_layer is not None:
            self.dropout_layer.build(h_shape)

        # FiLM expects (h, t_emb)
        self.film.build((h_shape, cond_shape))

        if self.norm_layer is not None:
            self.norm_layer.build(h_shape)

        self.act.build(h_shape)

        if self.residual:
            self.projector.units = h_shape[-1]
            self.projector.build(x_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, _ = input_shape
        if self.residual:
            return self.dense.compute_output_shape(x_shape)
        return self.dense.compute_output_shape(x_shape)

    def call(self, inputs, training=None, mask=None):
        x, cond = inputs

        h = x
        if self.norm_layer is not None:  # pre-normalization
            h = self.norm_layer(h, training=training)

        h = self.act(h)
        h = self.dense(h)

        # Inject condition via FiLM at this hidden layer
        h = self.film(h, cond)

        if self.dropout_layer is not None:
            h = self.dropout_layer(h, training=training)

        if self.residual:
            return self.projector(x) + h
        return h
