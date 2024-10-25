import keras
from keras import layers
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.utils import keras_kwargs
from bayesflow.utils import find_pooling


@serializable(package="bayesflow.networks")
class InvariantModule(keras.Layer):
    """Implements an invariant module performing a permutation-invariant transform.

    For details and rationale, see:

    [1] Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf
    """

    def __init__(
        self,
        num_dense_inner: int = 2,
        num_dense_outer: int = 2,
        units_inner: int = 128,
        units_outer: int = 128,
        activation: str = "gelu",
        kernel_initializer: str = "he_normal",
        dropout: int | float | None = 0.05,
        pooling: str | keras.Layer = "mean",
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
        self.inner_fc = keras.Sequential(name="InvariantInnerFC")
        for _ in range(num_dense_inner):
            layer = layers.Dense(
                units=units_inner,
                activation=activation,
                kernel_initializer=kernel_initializer,
            )
            if spectral_normalization:
                layer = layers.SpectralNormalization(layer)
            self.inner_fc.add(layer)

        # Outer fully connected net for sum decomposition: inner( pooling( inner(set) ) )
        self.outer_fc = keras.Sequential(name="InvariantOuterFC")
        for _ in range(num_dense_outer):
            if dropout is not None and dropout > 0:
                self.outer_fc.add(layers.Dropout(float(dropout)))

            layer = layers.Dense(
                units=units_outer,
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

    def call(self, input_set: Tensor, **kwargs) -> Tensor:
        """Performs the forward pass of a learnable invariant transform.

        Parameters
        ----------
        input_set : tf.Tensor
            Input of shape (batch_size,..., input_dim)

        Returns
        -------
        set_summary : tf.Tensor
            Output of shape (batch_size,..., out_dim)
        """

        set_summary = self.inner_fc(input_set, training=kwargs.get("training", False))
        set_summary = self.pooling_layer(set_summary, training=kwargs.get("training", False))
        set_summary = self.outer_fc(set_summary, training=kwargs.get("training", False))
        return set_summary
