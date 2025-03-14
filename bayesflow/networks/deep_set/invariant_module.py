from collections.abc import Sequence

import keras
from keras import layers
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.utils import find_pooling
from bayesflow.utils.decorators import sanitize_input_shape


@serializable(package="bayesflow.networks")
class InvariantModule(keras.Layer):
    """Implements an invariant module performing a permutation-invariant transform.

    For details and rationale, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(
        self,
        mlp_widths_inner: Sequence[int] = (128, 128),
        mlp_widths_outer: Sequence[int] = (128, 128),
        activation: str = "gelu",
        kernel_initializer: str = "he_normal",
        dropout: int | float | None = 0.05,
        pooling: str = "mean",
        pooling_kwargs: dict = None,
        spectral_normalization: bool = False,
    ):
        """
        Initializes an invariant module representing a learnable permutation-invariant function with an option for
        learnable pooling.

        This module applies a two-stage transformation: an inner fully connected network processes individual
        set elements, followed by a pooling operation that aggregates features across the set. The pooled features are
        then passed through an outer fully connected network to produce the final invariant representation.

        The model supports different activation functions, dropout, and optional spectral normalization for stability.
        The pooling mechanism can be customized with additional arguments.

        Parameters
        ----------
        mlp_widths_inner : Sequence[int], optional
            Widths of the MLP layers applied before pooling. Default is (128, 128).
        mlp_widths_outer : Sequence[int], optional
            Widths of the MLP layers applied after pooling. Default is (128, 128).
        activation : str, optional
            Activation function applied in the MLP layers, such as "gelu". Default is "gelu".
        kernel_initializer : str, optional
            Initialization strategy for kernel weights, such as "he_normal". Default is "he_normal".
        dropout : int, float, or None, optional
            Dropout rate applied in the outer MLP layers. Default is 0.05.
        pooling : str, optional
            Type of pooling operation applied across set elements, such as "mean". Default is "mean".
        pooling_kwargs : dict, optional
            Additional keyword arguments for the pooling layer. Default is None.
        spectral_normalization : bool, optional
            Whether to apply spectral normalization to stabilize training. Default is False.
        """

        super().__init__()

        # Inner fully connected net for sum decomposition: inner( pooling( inner(set) ) )
        self.inner_fc = keras.Sequential()
        for width in mlp_widths_inner:
            layer = layers.Dense(
                units=width,
                activation=activation,
                kernel_initializer=kernel_initializer,
            )
            if spectral_normalization:
                layer = layers.SpectralNormalization(layer)
            self.inner_fc.add(layer)

        # Outer fully connected net for sum decomposition: inner( pooling( inner(set) ) )
        self.outer_fc = keras.Sequential()
        for width in mlp_widths_outer:
            if dropout is not None and dropout > 0:
                self.outer_fc.add(layers.Dropout(float(dropout)))

            layer = layers.Dense(
                units=width,
                activation=activation,
                kernel_initializer=kernel_initializer,
            )
            if spectral_normalization:
                layer = layers.SpectralNormalization(layer)
            self.outer_fc.add(layer)

        # Pooling function as keras layer for sum decomposition: inner( pooling( inner(set) ) )
        if pooling_kwargs is None:
            pooling_kwargs = {}

        self.pooling_layer = find_pooling(pooling, **pooling_kwargs)

    @sanitize_input_shape
    def build(self, input_shape):
        self.call(keras.ops.zeros(input_shape))

    def call(self, input_set: Tensor, training: bool = False, **kwargs) -> Tensor:
        """Performs the forward pass of a learnable invariant transform.

        Parameters
        ----------
        input_set : Tensor
            Input of shape (batch_size,..., input_dim)
        training  : bool, optional, default - False
            Dictates the behavior of the optional dropout layers

        Returns
        -------
        set_summary : tf.Tensor
            Output of shape (batch_size,..., out_dim)
        """

        set_summary = self.inner_fc(input_set, training=training)
        set_summary = self.pooling_layer(set_summary, training=training)
        set_summary = self.outer_fc(set_summary, training=training)
        return set_summary
