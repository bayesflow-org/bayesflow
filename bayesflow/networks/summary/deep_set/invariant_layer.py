from collections.abc import Sequence

import keras

from bayesflow.types import Tensor
from bayesflow.utils import find_pooling, layer_kwargs
from bayesflow.utils.serialization import serializable

from ...subnets import MLP


@serializable("bayesflow.networks")
class InvariantLayer(keras.Layer):
    """Implements a permutation-invariant transform with Pre-LN RMSNorm.

    Applies the following transformation (no residual — pooling changes the set dimension):

        x → RMSNorm → inner MLP → pool → RMSNorm → outer MLP → output

    For details and rationale of the basic ideas, see:

    Bloem-Reddy, B., & Teh, Y. W. (2020). Probabilistic Symmetries and Invariant Neural Networks.
    J. Mach. Learn. Res., 21, 90-1. https://www.jmlr.org/papers/volume21/19-322/19-322.pdf

    Parameters
    ----------
    embed_dim : int, optional
        Output dimensionality and the working width of both MLPs. Default is 128.
    mlp_widths : Sequence[int], optional
        Hidden layer widths for both the inner and outer MLPs. The final
        output of each MLP is always ``embed_dim``. Default is ``(128,)``.
    pooling : str, optional
        Pooling operation applied after the inner MLP, e.g. ``"mean"``.
        Default is ``"mean"``.
    activation : str, optional
        Activation function for MLP layers. Default is ``"gelu"``.
    kernel_initializer : str, optional
        Weight initializer for Dense projections. Default is ``"he_normal"``.
    dropout : float or None, optional
        Dropout rate in MLP layers. Default is 0.05.
    layer_norm : bool, optional
        Whether to apply Pre-LN RMSNorm before each MLP. Default is True.
    **kwargs
        Additional keyword arguments forwarded to ``keras.Layer``.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        mlp_widths: Sequence[int] = (128,),
        pooling: str = "mean",
        activation: str = "gelu",
        kernel_initializer: str = "he_normal",
        dropout: float | None = 0.05,
        layer_norm: bool = True,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        self.embed_dim = embed_dim
        self.ln_inner = keras.layers.RMSNormalization() if layer_norm else None
        self.inner_fc = MLP(
            widths=(*mlp_widths, embed_dim),
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
        )

        self.ln_outer = keras.layers.RMSNormalization() if layer_norm else None
        self.outer_fc = MLP(
            widths=(*mlp_widths, embed_dim),
            activation=activation,
            kernel_initializer=kernel_initializer,
            dropout=dropout,
        )

        self.pooling_layer = find_pooling(pooling)

    def call(self, x: Tensor, training: bool = False) -> Tensor:
        """Performs the forward pass of a learnable invariant transform.

        Parameters
        ----------
        x : Tensor
            Input of shape ``(batch_size, set_size, input_dim)``.
        training : bool, optional
            Passed to dropout and norm layers, by default False.

        Returns
        -------
        Tensor
            Output of shape ``(batch_size, embed_dim)``.
        """
        if self.ln_inner is not None:
            x = self.ln_inner(x, training=training)
        x = self.inner_fc(x, training=training)
        x = self.pooling_layer(x)

        if self.ln_outer is not None:
            x = self.ln_outer(x, training=training)
        x = self.outer_fc(x, training=training)

        return x

    def build(self, input_shape):
        input_shape = tuple(input_shape)

        if self.ln_inner is not None:
            self.ln_inner.build(input_shape)
        self.inner_fc.build(input_shape)

        # pooling removes the set dimension (axis=-2): (B, S, embed_dim) → (B, embed_dim)
        pooled_shape = (input_shape[0], self.embed_dim)

        if self.ln_outer is not None:
            self.ln_outer.build(pooled_shape)
        self.outer_fc.build(pooled_shape)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.embed_dim)
