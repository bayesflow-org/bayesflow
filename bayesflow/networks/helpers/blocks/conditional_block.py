from typing import Literal, Callable

import keras

from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ..embeddings import FiLM


@serializable("bayesflow.networks")
class ConditionalBlock(keras.Layer):
    """Single fully-connected hidden layer with FiLM conditioning [1],
    optional residual skip, dropout, normalization, and spectral
    normalization.

    Expects a tuple ``(x, conditioning)`` as input.  The dense
    transformation is applied to ``x``, and ``conditioning`` modulates the
    hidden representation via Feature-wise Linear Modulation (FiLM).

    Computes::

        h = dense(x)
        h = dropout(h)  # if dropout > 0
        h = activation(h)
        h = film(h, conditioning)
        h = projector(x) + h  # if residual
        h = norm(h)  # if norm is not None

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
        Normalization applied after the residual addition. Default is
        ``"layer"``.
    spectral_normalization : bool, optional
        Apply spectral normalization to the Dense kernel. Default is
        ``False``.
    **kwargs
        Extra keyword arguments forwarded to ``keras.Layer``.

    References
    ----------
    [1] Perez et al. (2018), FiLM: Visual Reasoning with a General
        Conditioning Layer.
    """

    def __init__(
        self,
        width: int,
        *,
        activation: str | Callable[[], keras.Layer] = "mish",
        kernel_initializer: str | keras.Initializer = "he_normal",
        residual: bool = True,
        dropout: Literal[0, None] | float = 0.05,
        norm: Literal["batch", "layer"] | keras.Layer = "layer",
        spectral_normalization: bool = False,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        self.width = int(width)
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.residual = bool(residual)
        self.dropout = dropout
        self.norm = norm
        self.spectral_normalization = bool(spectral_normalization)

        # Dense
        dense = keras.layers.Dense(self.width, kernel_initializer=kernel_initializer, name="dense")
        if spectral_normalization:
            dense = keras.layers.SpectralNormalization(dense)
        self.dense = dense

        # Dropout
        self.dropout_layer = None
        if dropout is not None and dropout > 0:
            self.dropout_layer = keras.layers.Dropout(dropout, name="dropout")

        # Activation
        act = keras.activations.get(activation)
        if not isinstance(act, keras.Layer):
            act = keras.layers.Activation(act)
        self.act = act

        # FiLM
        self.film = FiLM(self.width, use_gamma=kwargs.pop("film_use_gamma", False), name="film")

        # Normalization
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

        # Residual projector (created in build if dims differ)
        self.projector = None

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
            "spectral_normalization": self.spectral_normalization,
        }
        return base | serialize(config)

    def build(self, input_shape):
        if self.built:
            return

        x_shape, cond_shape = input_shape

        self.dense.build(x_shape)
        h_shape = self.dense.compute_output_shape(x_shape)

        self.film.build((h_shape, cond_shape))

        if self.dropout_layer is not None:
            self.dropout_layer.build(h_shape)
        self.act.build(h_shape)

        if self.norm_layer is not None:
            self.norm_layer.build(h_shape)

        if self.residual and x_shape[-1] != h_shape[-1]:
            self.projector = keras.layers.Dense(
                h_shape[-1], kernel_initializer=self.kernel_initializer, name="projector"
            )
            self.projector.build(x_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, _ = input_shape
        return self.dense.compute_output_shape(x_shape)

    def call(self, inputs, training=None):
        x, cond = inputs
        h = self.dense(x)
        h = self.act(h)

        if self.dropout_layer is not None:
            h = self.dropout_layer(h, training=training)

        h = self.film((h, cond))

        if self.residual:
            skip = x if self.projector is None else self.projector(x)
            h = skip + h

        if self.norm_layer is not None:
            h = self.norm_layer(h, training=training)
        return h
