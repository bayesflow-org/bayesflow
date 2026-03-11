from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from .attention import SelfAttention2D
from .time_dense import TimeDense2D


@serializable("bayesflow.networks")
class TransformerBlock2D(keras.Layer):
    """
    Transformer block for NHWC feature maps (B,H,W,C):
        .. code-block:: text

            x = x + MLP(x, emb)
            x = x + Attn(x)
            or
            x = x + Attn(x)
            x = x + MLP(x, emb) # if mlp_first=False

    Notes
    -----
    - No residual rescaling (/sqrt(2)) here (matches simple diffusion transformer blocks).
    - Stability comes from zero-init output projections in the sub-blocks.

    [1] Hoogeboom et al. (2023), simple diffusion: End-to-end diffusion for high-resolution images

    Parameters
    ----------
    width : int
        Channel width `C` of the NHWC feature map. The MLP sub-block maps `C -> width -> C`.
    num_heads : int, optional
        Number of attention heads used by the self-attention sub-block.
    mlp_first : bool, optional
        If True, applies `MLP -> Attn`. If False, applies `Attn -> MLP`.
    attn_norm : {"group", "layer"}, optional
        Normalization type used inside the attention sub-block.
    attn_norm_with_bias : bool, optional
        Whether the attention normalization layer uses a learnable bias/offset.
    attn_groups : int, optional
        Number of groups for group normalization in the attention sub-block (ignored if `attn_norm="layer"`).
    attn_qk_norm : bool, optional
        Whether to apply QK normalization inside attention (often improves stability).
    attn_dropout : float or None, optional
        Dropout rate applied to attention weights / outputs (None treated as 0.0).
    mlp_activation : str, optional
        Activation function used in the time-conditioned MLP sub-block.
    mlp_norm : {"layer", "group"}, optional
        Normalization type used inside the MLP sub-block.
    mlp_norm_with_bias : bool, optional
        Whether the MLP normalization layer uses a learnable bias/offset.
    mlp_groups : int or None, optional
        Number of groups for group normalization in the MLP sub-block (ignored if `mlp_norm="layer"`).
    mlp_dropout : float or None, optional
        Dropout rate used inside the MLP sub-block (None treated as 0.0).
    mlp_use_film : bool, optional
        Whether the time embedding is FiLM (scale/shift) or additive injected in the MLP sub-block.
    kernel_initializer : str or keras.Initializer, optional
        Initializer used for kernels in both sub-blocks.
    **kwargs
        Additional keyword arguments forwarded to `keras.Layer`.
    """

    def __init__(
        self,
        width: int,
        *,
        num_heads: int = 4,
        mlp_first: bool = True,
        attn_norm: Literal["group", "layer"] = "group",
        attn_norm_with_bias: bool = False,
        attn_groups: int = 8,
        attn_qk_norm: bool = True,
        attn_dropout: float | None = 0.0,
        mlp_activation: str = "mish",
        mlp_norm: Literal["layer", "group"] = "group",
        mlp_norm_with_bias: bool = False,
        mlp_groups: int | None = 8,
        mlp_dropout: float | None = 0.0,
        mlp_use_film: bool = True,
        kernel_initializer: str | keras.Initializer = "he_normal",
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        self.width = width
        self.num_heads = num_heads
        self.mlp_first = mlp_first

        self.attn_norm = attn_norm
        self.attn_norm_with_bias = attn_norm_with_bias
        self.attn_groups = attn_groups
        self.attn_qk_norm = attn_qk_norm
        self.attn_dropout = 0.0 if attn_dropout is None else attn_dropout

        self.mlp_activation = mlp_activation
        self.mlp_norm = mlp_norm
        self.mlp_norm_with_bias = mlp_norm_with_bias
        self.mlp_groups = mlp_groups
        self.mlp_dropout = 0.0 if mlp_dropout is None else mlp_dropout
        self.mlp_use_film = mlp_use_film

        self.kernel_initializer = kernel_initializer

        # IMPORTANT: residual="none" in sublayers; residual add happens here.
        self.attn = SelfAttention2D(
            num_heads=self.num_heads,
            norm=self.attn_norm,
            norm_with_bias=self.attn_norm_with_bias,
            groups=self.attn_groups,
            qk_norm=self.attn_qk_norm,
            attn_dropout=self.attn_dropout,
            residual="none",
            kernel_initializer=self.kernel_initializer,
        )

        self.mlp = TimeDense2D(
            width=self.width,
            activation=self.mlp_activation,
            norm=self.mlp_norm,
            norm_with_bias=self.mlp_norm_with_bias,
            groups=self.mlp_groups,
            dropout=self.mlp_dropout,
            residual="none",
            kernel_initializer=self.kernel_initializer,
            use_film=self.mlp_use_film,
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "width": self.width,
            "num_heads": self.num_heads,
            "mlp_first": self.mlp_first,
            "attn_norm": self.attn_norm,
            "attn_norm_with_bias": self.attn_norm_with_bias,
            "attn_groups": self.attn_groups,
            "attn_qk_norm": self.attn_qk_norm,
            "attn_dropout": self.attn_dropout,
            "mlp_activation": self.mlp_activation,
            "mlp_norm": self.mlp_norm,
            "mlp_norm_with_bias": self.mlp_norm_with_bias,
            "mlp_groups": self.mlp_groups,
            "mlp_dropout": self.mlp_dropout,
            "mlp_use_film": self.mlp_use_film,
            "kernel_initializer": self.kernel_initializer,
        }
        return base_config | serialize(config)

    def build(self, input_shape):
        if self.built:
            return

        x_shape, emb_shape = input_shape
        self.attn.build(x_shape)
        self.mlp.build((x_shape, emb_shape))

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, _ = input_shape
        return x_shape

    def call(
        self,
        inputs: tuple[Tensor, Tensor],
        training: bool | None = None,
        **kwargs,
    ) -> Tensor:
        x, emb = inputs
        if self.mlp_first:
            x = x + self.mlp((x, emb), training=training)
            x = x + self.attn(x, training=training)
        else:
            x = x + self.attn(x, training=training)
            x = x + self.mlp((x, emb), training=training)
        return x
