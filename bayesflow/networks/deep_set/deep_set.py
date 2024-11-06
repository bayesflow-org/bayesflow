import keras
from keras import layers
from keras.saving import register_keras_serializable as serializable, serialize_keras_object as serialize


from bayesflow.types import Tensor
from .invariant_module import InvariantModule
from .equivariant_module import EquivariantModule

from ..summary_network import SummaryNetwork


@serializable(package="bayesflow.networks")
class DeepSet(SummaryNetwork):
    r"""Implements a deep set encoder introduced in [1]. This module performs the computation:

    ..math:
        f(X = \{ x_i \mid i=1, \ldots, n \}) = \rho \left( \sigma(\tau(x_1), \ldots, \tau(x_n)) \right)

    where $\sigma must be a permutation-invariant function, such as the mean.
    $\rho$ and $\tau$ can be any functions, such as neural networks.

    [1] Deep Set: arXiv:1703.06114
    """

    def __init__(
        self,
        summary_dim: int = 16,
        depth: int = 2,
        inner_pooling: str = "mean",
        output_pooling: str = "mean",
        mlp_widths_equivariant: Sequence[int] = (128, 128),
        mlp_widths_invariant_inner: Sequence[int] = (128, 128),
        mlp_widths_invariant_outer: Sequence[int] = (128, 128),
        mlp_widths_invariant_last: Sequence[int] = (128, 128),
        activation: str = "gelu",
        kernel_initializer: str = "he_normal",
        dropout: int | float | None = 0.05,
        spectral_normalization: bool = False,
        **kwargs,
    ):
        """
        #TODO
        """

        super().__init__(**kwargs)

        # Stack of equivariant modules for a many-to-many learnable transformation
        self.equivariant_modules = keras.Sequential()
        for _ in range(depth):
            equivariant_module = EquivariantModule(
                mlp_widths_equivariant=mlp_widths_equivariant,
                mlp_widths_invariant_inner=mlp_widths_invariant_inner,
                mlp_widths_invariant_outer=mlp_widths_invariant_outer,
                activation=activation,
                kernel_initializer=kernel_initializer,
                spectral_normalization=spectral_normalization,
                dropout=dropout,
                pooling=inner_pooling,
                **kwargs,
            )
            self.equivariant_modules.add(equivariant_module)

        # Invariant module for a many-to-one transformation
        self.invariant_module = InvariantModule(
            mlp_widths_inner=mlp_widths_invariant_last,
            mlp_widths_outer=mlp_widths_invariant_last,
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
            pooling=output_pooling,
            spectral_normalization=spectral_normalization,
            **kwargs,
        )

        # Output linear layer to project set representation down to "summary_dim" learned summary statistics
        self.output_projector = layers.Dense(summary_dim, activation="linear")
        self.summary_dim = summary_dim

    def build(self, input_shape):
        super().build(input_shape)
        self.call(keras.ops.zeros(input_shape))

    def call(self, input_set: Tensor, training: bool = False, **kwargs) -> Tensor:
        """Performs the forward pass of a learnable deep invariant transformation consisting of
        a sequence of equivariant transforms followed by an invariant transform.

        #TODO
        """
        x = self.equivariant_modules(input_set, training=training)
        x = self.invariant_module(x, training=training)

        return self.output_projector(x)

    def get_config(self):
        base_config = super().get_config()

        config = {
            "invariant_module": serialize(self.equivariant_modules),
            "equivariant_fc": serialize(self.invariant_module),
            "output_projector": serialize(self.output_projector),
            "summary_dim": serialize(self.summary_dim),
        }

        return base_config | config
