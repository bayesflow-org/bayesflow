from collections.abc import Sequence

import keras
from keras import layers
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.utils import find_pooling
from bayesflow.utils import keras_kwargs


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
        spectral_normalization: bool = False,
        **kwargs,
    ):
        """Creates an invariant module according to [1] which represents a learnable permutation-invariant
        function with an option for learnable pooling.

        Parameters
        ----------
        # TODO

        **kwargs: dict
            Optional keyword arguments can be passed to the pooling layer as a dictionary into the
            reserved key ``pooling_kwargs``. Example: #TODO
        """
        super().__init__(**keras_kwargs(kwargs))

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
        # TODO: why does using Sequential work here, but not in DeepSet?
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
        self.pooling_layer = find_pooling(pooling, **kwargs.get("pooling_kwargs", {}))

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
