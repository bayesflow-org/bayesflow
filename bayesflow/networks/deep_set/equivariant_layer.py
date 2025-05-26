from collections.abc import Sequence

import keras
from keras import ops, layers

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.decorators import sanitize_input_shape
from bayesflow.utils.serialization import serializable

from ..mlp import MLP

from .invariant_layer import InvariantLayer


@serializable("bayesflow.networks")
class EquivariantLayer(keras.Layer):
    """Implements an equivariant module performing an equivariant transform.

    For details and justification, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(
        self,
        mlp_widths_equivariant: Sequence[int] = (128, 128),
        mlp_widths_invariant_inner: Sequence[int] = (128, 128),
        mlp_widths_invariant_outer: Sequence[int] = (128, 128),
        pooling: str = "mean",
        activation: str = "gelu",
        kernel_initializer: str = "he_normal",
        dropout: int | float | None = 0.05,
        layer_norm: bool = True,
        spectral_normalization: bool = False,
        **kwargs,
    ):
        """
        Initializes an equivariant module that combines equivariant transformations with nested invariant transforms
        to enable interactions between set members.

        This module applies an equivariant transformation to each set member, followed by an invariant transformation
        that aggregates information and injects it back into the set representation.

        The architecture consists of a fully connected residual block for equivariant processing and an invariant
        module to enhance expressiveness.

        The model supports different activation functions, dropout, layer normalization, and optional spectral
        normalization for stability.

        Parameters
        ----------
        mlp_widths_equivariant : Sequence[int], optional
            Widths of the MLP layers in the equivariant transformation applied to each set member.
            Default is (128, 128).
        mlp_widths_invariant_inner : Sequence[int], optional
            Widths of the inner MLP layers in the invariant module. Default is (128, 128).
        mlp_widths_invariant_outer : Sequence[int], optional
            Widths of the outer MLP layers in the invariant module. Default is (128, 128).
        pooling : str, optional
            Type of pooling operation used in the invariant module, such as "mean". Default is "mean".
        activation : str, optional
            Activation function applied in the MLP layers, such as "gelu". Default is "gelu".
        kernel_initializer : str, optional
            Initialization strategy for kernel weights, such as "he_normal". Default is "he_normal".
        dropout : int, float, or None, optional
            Dropout rate applied within the MLP layers. Default is 0.05.
        layer_norm : bool, optional
            Whether to apply layer normalization after transformations. Default is True.
        spectral_normalization : bool, optional
            Whether to apply spectral normalization to stabilize training. Default is False.
        """

        super().__init__(**layer_kwargs(kwargs))

        # Invariant module to increase expressiveness by concatenating outputs to each set member
        self.invariant_module = InvariantLayer(
            mlp_widths_inner=mlp_widths_invariant_inner,
            mlp_widths_outer=mlp_widths_invariant_outer,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            pooling=pooling,
            spectral_normalization=spectral_normalization,
        )

        # Fully connected net + residual connection for an equivariant transform applied to each set member
        self.input_projector = layers.Dense(mlp_widths_equivariant[-1])
        self.equivariant_fc = MLP(
            mlp_widths_equivariant,
            dropout=dropout,
            activation=activation,
            kernel_initializer=kernel_initializer,
            spectral_normalization=spectral_normalization,
        )
        self.out_fc_projector = keras.layers.Dense(mlp_widths_equivariant[-1], kernel_initializer=kernel_initializer)

        self.layer_norm = layers.LayerNormalization() if layer_norm else None

    @sanitize_input_shape
    def build(self, input_shape):
        self.call(keras.ops.zeros(input_shape))

    def call(self, input_set: Tensor, training: bool = False, **kwargs) -> Tensor:
        """Performs the forward pass of a learnable equivariant transform.

        Parameters
        ----------
        input_set : Tensor
            The input tensor representing a set, with shape
            (batch_size, ..., set_size, input_dim).
        training : bool, optional
            A flag indicating whether the model is in training mode. Default is False.
        **kwargs : dict
            Additional keyword arguments for compatibility with other functions.

        Returns
        -------
        Tensor
            The transformed output tensor with the same shape as `input_set`, where
            each element is processed through the equivariant transformation.
        """

        input_set = self.input_projector(input_set)

        # Store shape of input_set, will be (batch_size, ..., set_size, some_dim)
        shape = ops.shape(input_set)

        # Example: Output dim is (batch_size, ..., set_size, representation_dim)
        invariant_summary = self.invariant_module(input_set, training=training)
        invariant_summary = ops.expand_dims(invariant_summary, axis=-2)
        tiler = [1] * len(shape)
        tiler[-2] = shape[-2]
        invariant_summary = ops.tile(invariant_summary, tiler)

        # Concatenate each input entry with the repeated invariant embedding
        output_set = ops.concatenate([input_set, invariant_summary], axis=-1)

        # Pass through final equivariant transform + residual
        out_fc = self.equivariant_fc(output_set, training=training)
        out_projected = self.out_fc_projector(out_fc)
        output_set = input_set + out_projected

        if self.layer_norm is not None:
            output_set = self.layer_norm(output_set, training=training)

        return output_set
