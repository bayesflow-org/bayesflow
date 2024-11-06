from collections.abc import Sequence
from typing import Literal

import keras
from keras import layers
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.utils import keras_kwargs
from .hidden_block import ConfigurableHiddenBlock


@serializable(package="bayesflow.networks")
class MLP(keras.Layer):
    """
    Implements a simple configurable MLP with optional residual connections and dropout.

    If used in conjunction with a coupling net, a diffusion model, or a flow matching model, it assumes
    that the input and conditions are already concatenated (i.e., this is a single-input model).
    """

    def __init__(
        self,
        *,
        depth: int = None,
        width: int = None,
        widths: Sequence[int] = None,
        activation: str = "mish",
        kernel_initializer: str = "he_normal",
        residual: bool = True,
        dropout: Literal[0, None] | float = 0.05,
        spectral_normalization: bool = False,
        **kwargs,
    ):
        """
        Creates an instance of a flexible and simple MLP with optional residual connections and dropout.

        Parameters:
        -----------
        widths           : tuple, optional, default: (512, 512)
            The number of hidden units for each (residual) hidden layer.
            Note: The depth of the network is inferred from len(widths)
        activation       : string, optional, default: 'gelu'
            The activation function of the dense layers
        residual         : bool, optional, default: True
            Use residual connections in the internal layers.
        spectral_normalization    : bool, optional, default: False
            Use spectral normalization for the network weights, which can make
            the learned function smoother and hence more robust to perturbations.
        dropout          : float, optional, default: 0.05
            Dropout rate for the hidden layers in the internal layers.
        """
        super().__init__(**keras_kwargs(kwargs))

        if widths is not None:
            if depth is not None or width is not None:
                raise ValueError("Either specify 'widths' or 'depth' and 'width', not both.")
        else:
            if depth is None or width is None:
                # use the default
                depth = 2
                width = 512

            widths = [width] * depth

        self.res_blocks = []
        projector = layers.Dense(
            units=widths[0],
            kernel_initializer=kernel_initializer,
        )
        if spectral_normalization:
            projector = layers.SpectralNormalization(projector)
        self.res_blocks.append(projector)

        if dropout is not None and dropout > 0.0:
            self.res_blocks.append(layers.Dropout(float(dropout)))

        for width in widths:
            self.res_blocks.append(
                ConfigurableHiddenBlock(
                    units=width,
                    activation=activation,
                    kernel_initializer=kernel_initializer,
                    residual=residual,
                    dropout=dropout,
                    spectral_normalization=spectral_normalization,
                )
            )

    def build(self, input_shape):
        for layer in self.res_blocks:
            layer.build(input_shape)
            input_shape = layer.compute_output_shape(input_shape)

    def call(self, x: Tensor, training: bool = False, **kwargs) -> Tensor:
        for layer in self.res_blocks:
            x = layer(x, training=training)
        return x

    def compute_output_shape(self, input_shape):
        for layer in self.res_blocks:
            input_shape = layer.compute_output_shape(input_shape)

        return input_shape
