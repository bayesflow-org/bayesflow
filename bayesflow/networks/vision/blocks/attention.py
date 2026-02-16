from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from bayesflow.networks.vision.blocks.norms import SimpleNorm


@serializable("bayesflow.networks")
class SelfAttention2D(keras.Layer):
    """
    Self-attention block for NHWC feature maps (B, H, W, C).

    This layer follows the common diffusion U-Net attention pattern [1, 2]:
      - Normalize input (often GroupNorm in U-Nets)
      - Compute Q, K, V using linear projections
      - (Optional) normalize Q and K with bias (GroupNorm-like) as in [2].
      - Dot-product attention over spatial positions (HW tokens)
      - Final projection uses zero-init so the block starts near-identity when used with residual adds

    Notes:
        - This layer does not change the channel dimension; it infers C at build time from the input.
        - Differences between [1] and [2] are (single head; no Q/K norm) and (multi-head; Q/K norms) explicitly.

    [1] Nain (2022) Keras example: Denoising Diffusion Probabilistic Model (https://keras.io/examples/generative/ddpm/)

    [2] Hoogeboom et al. (2023), simple diffusion: End-to-end diffusion for high-resolution images

    """

    def __init__(
        self,
        *,
        num_heads: int = 4,
        norm: Literal["group", "layer"] = "group",
        norm_with_bias: bool = False,
        groups: int = 8,
        qk_norm: bool = True,
        qk_norm_type: Literal["layer"] = "layer",
        attn_dropout: Literal[0, None] | float = 0.0,
        residual: Literal["none", "input", "norm"] | None = "none",
        kernel_initializer: str | keras.Initializer = "he_normal",
        **kwargs,
    ):
        """
        Parameters
        ----------
        num_heads : int, optional
            Number of attention heads. Set to 1 for single-head attention. Default is 4.
        norm : {"group", "layer"}, optional
            Normalization applied to the input before projecting Q/K/V. Default is "group".
        norm_with_bias : bool, optional
            If True, applies a bias term in the input normalization (e.g., GroupNorm with center=True) to match the
            pattern from [1]. Default is False since as in [2].
        groups : int, optional
            Number of groups for GroupNorm. Ignored for LayerNorm. If given, may be adjusted in build()
            by SimpleNorm to divide C. Default is 8.
        qk_norm : bool, optional
            If True, applies LayerNorm-with-bias to Q and K (per-head, over head_dim), matching the
            "NormalizeWithBias(Q/K)" pattern. Default is True.
        attn_dropout : float or None, optional
            Dropout applied to attention weights after softmax. Default is 0.0.
        residual : {"none", "input", "norm"}, optional
            Controls what the layer returns:
              - "none": returns only the attention projection (delta) with shape (B,H,W,C)
              - "input": returns x + delta (residual on raw input)
              - "norm": returns input_norm(x) + delta (matches Keras DDPM example [2]: returns normalized inputs + proj)
            Default is "none" (best for `x += attn(x)`).
        kernel_initializer : str or keras.Initializer, optional
            Initializer for Q/K/V projections. The output projection is zero-initialized.
        **kwargs
            Additional keyword arguments passed to `keras.Layer`.
        """
        super().__init__(**layer_kwargs(kwargs))
        self.num_heads = int(num_heads)

        self.norm = str(norm)
        self.groups = int(groups)
        self.norm_with_bias = bool(norm_with_bias)
        self.qk_norm = bool(qk_norm)
        self.qk_norm_type = str(qk_norm_type)
        self.attn_dropout = 0.0 if attn_dropout is None else float(attn_dropout)
        self.residual = "none" if residual is None else str(residual)
        self.kernel_initializer = kernel_initializer



        # Input norm (pre-norm attention)
        self.x_norm = SimpleNorm(
            method="group" if self.norm == "group" else "layer",
            groups=self.groups,
            axis=-1,
            center=self.norm_with_bias,
            scale=True,
            name="x_norm",
        )

        # set at build time when channel_dim is known
        self.channel_dim = None
        self.head_dim = None
        self.q_proj = None
        self.k_proj = None
        self.v_proj = None
        self.out_proj = None

        # Optional Q/K normalization ("NormalizeWithBias")
        if self.qk_norm:
            if self.qk_norm_type != "layer":
                raise ValueError("SelfAttention2D currently only supports qk_norm='layer'.")
            # normalize over head_dim (last axis of (B, HW, heads, head_dim))
            self.q_norm = SimpleNorm(
                method="layer",
                axis=-1,
                center=True,
                scale=True,
                name="q_norm",
            )
            self.k_norm = SimpleNorm(
                method="layer",
                axis=-1,
                center=True,
                scale=True,
                name="k_norm",
            )
        else:
            self.q_norm = None
            self.k_norm = None

        self.softmax = keras.layers.Softmax(axis=-1, name="attn_softmax")
        self.weights_dropout = keras.layers.Dropout(self.attn_dropout, name="attn_dropout")

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = layer_kwargs(super().get_config())
        cfg = {
            "num_heads": self.num_heads,
            "norm": self.norm,
            "norm_with_bias": self.norm_with_bias,
            "groups": self.groups,
            "qk_norm": self.qk_norm,
            "qk_norm_type": self.qk_norm_type,
            "attn_dropout": self.attn_dropout,
            "residual": self.residual,
            "kernel_initializer": self.kernel_initializer,
        }
        return base | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return

        # input_shape: (B, H, W, C)
        in_ch = input_shape[-1]
        if in_ch is None:
            raise ValueError("SelfAttention2D requires a known channel dimension.")
        self.channel_dim = int(in_ch)

        if self.channel_dim % self.num_heads != 0:
            raise ValueError(
                f"`num_heads` must divide `channel_dim`, but got num_heads={self.num_heads} for "
                f"channel_dim={self.channel_dim}."
            )
        self.head_dim = self.channel_dim // self.num_heads

        self.q_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            kernel_initializer=self.kernel_initializer,
            name="q_proj",
        )
        self.k_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            kernel_initializer=self.kernel_initializer,
            name="k_proj",
        )
        self.v_proj = keras.layers.Dense(
            self.num_heads * self.head_dim,
            kernel_initializer=self.kernel_initializer,
            name="v_proj",
        )
        # Output projection (zero-init like diffusion U-Nets)
        self.out_proj = keras.layers.Dense(
            self.channel_dim,
            kernel_initializer="zeros",
            name="out_proj_zero",
        )

        self.x_norm.build(input_shape)
        tok_shape = (None, None, self.channel_dim)
        self.q_proj.build(tok_shape)
        self.k_proj.build(tok_shape)
        self.v_proj.build(tok_shape)

        if self.qk_norm:
            # (B, HW, heads, head_dim)
            self.q_norm.build((None, None, self.num_heads, self.head_dim))
            self.k_norm.build((None, None, self.num_heads, self.head_dim))

        attn_shape = (None, self.num_heads, None, None)
        self.softmax.build(attn_shape)
        self.weights_dropout.build(attn_shape)

        self.out_proj.build(tok_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, x: Tensor, training: bool | None = None, **kwargs) -> Tensor:
        # x: (B, H, W, C)
        b = keras.ops.shape(x)[0]
        h = keras.ops.shape(x)[1]
        w = keras.ops.shape(x)[2]
        hw = h * w

        # pre-norm
        x_n = self.x_norm(x, training=training)

        # flatten to tokens: (B, HW, C)
        x_tok = keras.ops.reshape(x_n, (b, hw, self.channel_dim))

        # project Q/K/V: (B, HW, heads*head_dim) -> reshape (B, HW, heads, head_dim)
        q = self.q_proj(x_tok, training=training)
        k = self.k_proj(x_tok, training=training)
        v = self.v_proj(x_tok, training=training)

        q = keras.ops.reshape(q, (b, hw, self.num_heads, self.head_dim))
        k = keras.ops.reshape(k, (b, hw, self.num_heads, self.head_dim))
        v = keras.ops.reshape(v, (b, hw, self.num_heads, self.head_dim))

        if self.qk_norm:
            q = self.q_norm(q, training=training)
            k = self.k_norm(k, training=training)

        # scaled dot-product attention
        q = q * (self.head_dim ** -0.5)

        # (b=B, h=heads, q=HW, k=HW, d=head_dim): (B, heads, HW, HW)
        attn = keras.ops.einsum("bqhd,bkhd->bhqk", q, k)
        attn = self.softmax(attn)
        attn = self.weights_dropout(attn, training=training)

        # (B, HW, heads, head_dim)
        out = keras.ops.einsum("bhqk,bkhd->bqhd", attn, v)
        out = keras.ops.reshape(out, (b, hw, self.num_heads * self.head_dim))

        # output proj to C, then reshape back to (B, H, W, C)
        out = self.out_proj(out, training=training)
        out = keras.ops.reshape(out, (b, h, w, self.channel_dim))

        # Residual behavior control
        match self.residual:
            case "none":
                return out
            case "input":
                return x + out
            case "norm":
                return x_n + out
            case _:
                raise ValueError(f"Unknown residual mode: {self.residual}")
