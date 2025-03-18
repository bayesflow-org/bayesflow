from collections.abc import Sequence
from typing import Literal

import keras
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
        residual: bool = False,
        dropout: Literal[0, None] | float = 0.05,
        spectral_normalization: bool = False,
        **kwargs,
    ):
        """
        Implements a flexible multi-layer perceptron (MLP) with optional residual connections, dropout, and
        spectral normalization.

        This MLP can be used as a general-purpose feature extractor or function approximator,supporting configurable
        depth, width, activation functions, and weight initializations.

        If `residual` is enabled, each layer includes a skip connection for improved gradient flow. The model also
        supports dropout for regularization and spectral normalization for stability in learning smooth functions.

        The architecture can be specified either via an explicit sequence of layer widths (`widths`) or by defining a
        fixed depth and width (`depth` and `width`).

        Parameters
        ----------
        depth : int, optional
            Number of layers in the MLP when `widths` is not explicitly provided. Must be
            used together with `width`. Default is 2.
        width : int, optional
            Number of units per layer when `widths` is not explicitly provided. Must be used
            together with `depth`. Default is 256.
        widths : Sequence[int], optional
            Explicitly defines the number of hidden units per layer. If provided, `depth` and
            `width` should not be specified. Default is None.
        activation : str, optional
            Activation function applied in the hidden layers, such as "mish". Default is "mish".
        kernel_initializer : str, optional
            Initialization strategy for kernel weights, such as "he_normal". Default is "he_normal".
        residual : bool, optional
            Whether to use residual connections for improved training stability. Default is False.
        dropout : float or None, optional
            Dropout rate applied within the MLP layers for regularization. Default is 0.05.
        spectral_normalization : bool, optional
            Whether to apply spectral normalization to stabilize training. Default is False.
        **kwargs
            Additional keyword arguments passed to the Keras layer initialization.

        Raises
        ------
        ValueError
            If both `widths` and (`depth`, `width`) are provided
        """

        super().__init__(**keras_kwargs(kwargs))

        if widths is not None:
            if depth is not None or width is not None:
                raise ValueError("Either specify 'widths' or 'depth' and 'width', not both.")
        else:
            if depth is None or width is None:
                # use the default
                depth = 2
                width = 256

            widths = [width] * depth

        self.res_blocks = []
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
