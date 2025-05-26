from collections.abc import Sequence
from typing import Literal, Callable

import keras

from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize

from ..sequential import Sequential
from ..residual import Residual


@serializable("bayesflow.networks")
class MLP(Sequential):
    """
    Implements a simple configurable MLP with optional residual connections and dropout.

    If used in conjunction with a coupling net, a diffusion model, or a flow matching model, it assumes
    that the input and conditions are already concatenated (i.e., this is a single-input model).
    """

    def __init__(
        self,
        widths: Sequence[int] = (256, 256),
        *,
        activation: str | Callable[[], keras.Layer] = "mish",
        kernel_initializer: str | keras.Initializer = "he_normal",
        residual: bool = True,
        dropout: Literal[0, None] | float = 0.05,
        norm: Literal["batch", "layer"] | keras.Layer = None,
        spectral_normalization: bool = False,
        **kwargs,
    ):
        """
        Implements a flexible multi-layer perceptron (MLP) with optional residual connections, dropout, and
        spectral normalization.

        This MLP can be used as a general-purpose feature extractor or function approximator, supporting configurable
        depth, width, activation functions, and weight initializations.

        If `residual` is enabled, each layer includes a skip connection for improved gradient flow. The model also
        supports dropout for regularization and spectral normalization for stability in learning smooth functions.

        Parameters
        ----------
        widths : Sequence[int], optional
            Defines the number of hidden units per layer, as well as the number of layers to be used.
        activation : str, optional
            Activation function applied in the hidden layers, such as "mish". Default is "mish".
        kernel_initializer : str, optional
            Initialization strategy for kernel weights, such as "he_normal". Default is "he_normal".
        residual : bool, optional
            Whether to use residual connections for improved training stability. Default is False.
        dropout : float or None, optional
            Dropout rate applied within the MLP layers for regularization. Default is 0.05.
        norm: str, optional

        spectral_normalization : bool, optional
            Whether to apply spectral normalization to stabilize training. Default is False.
        **kwargs
            Additional keyword arguments passed to the Keras layer initialization.
        """
        self.widths = list(widths)
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.residual = residual
        self.dropout = dropout
        self.norm = norm
        self.spectral_normalization = spectral_normalization

        blocks = []

        for width in widths:
            block = self._make_block(
                width, activation, kernel_initializer, residual, dropout, norm, spectral_normalization
            )
            blocks.append(block)

        super().__init__(*blocks, **kwargs)

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "widths": self.widths,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "residual": self.residual,
            "dropout": self.dropout,
            "norm": self.norm,
            "spectral_normalization": self.spectral_normalization,
        }

        return base_config | serialize(config)

    @staticmethod
    def _make_block(width, activation, kernel_initializer, residual, dropout, norm, spectral_normalization):
        layers = []

        dense = keras.layers.Dense(width, kernel_initializer=kernel_initializer)
        if spectral_normalization:
            dense = keras.layers.SpectralNormalization(dense)
        layers.append(dense)

        if dropout is not None and dropout > 0:
            layers.append(keras.layers.Dropout(dropout))

        activation = keras.activations.get(activation)
        if not isinstance(activation, keras.Layer):
            activation = keras.layers.Activation(activation)

        layers.append(activation)

        if norm == "batch":
            layers.append(keras.layers.BatchNormalization())
        elif norm == "layer":
            layers.append(keras.layers.LayerNormalization())
        elif isinstance(norm, str):
            raise ValueError(f"Unknown normalization strategy: {norm!r}.")
        elif isinstance(norm, keras.Layer):
            layers.append(norm)
        elif norm is None:
            pass
        else:
            raise TypeError(f"Cannot infer norm from {norm!r} of type {type(norm)}.")

        if residual:
            return Residual(*layers)

        return Sequential(layers)
