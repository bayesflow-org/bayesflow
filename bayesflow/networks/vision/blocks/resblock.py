from __future__ import annotations

import math
from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from bayesflow.networks.vision.blocks.norms_simple import SimpleNorm


@serializable("bayesflow.networks")
class ResidualBlock2D(keras.Layer):
    """
    Time-conditioned residual block for image backbones (NHWC).

    The block conditions on a *global* embedding vector (e.g., time / log-SNR embedding / sinusoidal embedding).
    Spatial conditioning maps (H, W, Cc) are intentionally not handled here; the backbone
    wrapper should decide where/how often to inject spatial conditions.

    Optionally supports "simple diffusion"-style additive skip fusion:
        h = (Norm(x) + Norm(skip_h)) / sqrt(2)
    controlled via `skip_fuse="add_sqrt2"` (see https://arxiv.org/abs/2301.11093 for details.)
    """

    def __init__(
        self,
        width: int,
        *,
        activation: str = "swish",
        norm: Literal["layer", "group"] = "group",
        groups: int = 8,
        dropout: float = 0.0,
        kernel_initializer: str | keras.initializers.Initializer = "he_normal",
        use_scale_shift: bool = True,
        skip_fuse: Literal["none", "add_sqrt2"] = "none",
        epsilon: float = 1e-5,
        **kwargs,
    ):
        """
        Parameters
        ----------
        width : int
            Number of output channels produced by the block.
        activation : str, optional
            Activation function used inside the block. Default is "swish".
        norm : {"layer", "group"}, optional
            Normalization type. Default is "group".
        groups : int, optional
            Number of groups for group normalization. Adjusted in `build()` if needed
            to divide the channel dimension.
        dropout : float, optional
            Dropout rate applied before the second convolution. Default is 0.0.
        kernel_initializer : str or keras.initializers.Initializer, optional
            Initializer for conv1 and embedding projection. conv2 uses zero init to start
            near-identity (common in diffusion U-Nets).
        use_scale_shift : bool, optional
            If True, uses FiLM-style scale-and-shift from the embedding (AdaNorm).
            If False, uses additive embedding bias.
        skip_fuse : {"none", "add_sqrt2"}, optional
            If "add_sqrt2" and `skip_h` is passed at call time, fuses `x` and `skip_h` as
            (Norm(x) + Norm(skip_h)) / sqrt(2) before conv1.
        epsilon : float, optional
            Numerical stability constant for normalization layers. Default is 1e-5.
        **kwargs
            Additional keyword arguments passed to `keras.Layer`.
        """
        super().__init__(**layer_kwargs(kwargs))
        self.width = int(width)
        self.activation = str(activation)
        self.norm = str(norm)
        self.groups = int(groups)
        self.dropout = float(dropout)
        self.kernel_initializer = kernel_initializer
        self.use_scale_shift = bool(use_scale_shift)
        self.skip_fuse = str(skip_fuse)
        self.epsilon = float(epsilon)

        # --- layers ---
        self.norm1 = SimpleNorm(
            method=self.norm,
            groups=self.groups,
            center=True,
            scale=True,
            epsilon=self.epsilon,
            name="norm1",
        )
        self.act1 = keras.layers.Activation(self.activation, name="act1")
        self.conv1 = keras.layers.Conv2D(
            filters=self.width,
            kernel_size=3,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            name="conv1",
        )

        emb_out = 2 * self.width if self.use_scale_shift else self.width
        self.emb_proj = keras.layers.Dense(
            emb_out,
            kernel_initializer=self.kernel_initializer,
            name="emb_proj",
        )

        self.norm2 = SimpleNorm(
            method=self.norm,
            groups=self.groups,
            center=True,
            scale=True,
            epsilon=self.epsilon,
            name="norm2",
        )
        self.act2 = keras.layers.Activation(self.activation, name="act2")
        self.drop = keras.layers.Dropout(rate=self.dropout, name="dropout")
        self.conv2 = keras.layers.Conv2D(
            filters=self.width,
            kernel_size=3,
            padding="same",
            kernel_initializer="zeros",
            name="conv2_zero",
        )

        # Optional skip normalization (only used when skip_h is passed)
        self.skip_norm = SimpleNorm(
            method=self.norm,
            groups=self.groups,
            center=True,
            scale=True,
            epsilon=self.epsilon,
            name="skip_norm",
        )

        # residual projection if channels mismatch (created in build)
        self.res_proj: keras.Layer | None = None

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = layer_kwargs(super().get_config())
        cfg = {
            "width": self.width,
            "activation": self.activation,
            "norm": self.norm,
            "groups": self.groups,
            "dropout": self.dropout,
            "kernel_initializer": self.kernel_initializer,
            "use_scale_shift": self.use_scale_shift,
            "skip_fuse": self.skip_fuse,
            "epsilon": self.epsilon,
        }
        return base | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return

        x_shape, emb_shape = input_shape
        in_ch = x_shape[-1]

        # Residual projection (only if channel mismatch)
        if in_ch is not None and int(in_ch) != self.width:
            self.res_proj = keras.layers.Conv2D(
                filters=self.width,
                kernel_size=1,
                padding="same",
                kernel_initializer=self.kernel_initializer,
                name="res_proj",
            )
        else:
            self.res_proj = keras.layers.Identity(name="res_identity")

        # Build sublayers
        self.norm1.build(x_shape)
        self.act1.build(x_shape)
        self.conv1.build(x_shape)
        h_shape = self.conv1.compute_output_shape(x_shape)

        self.emb_proj.build(emb_shape)

        self.norm2.build(h_shape)
        self.act2.build(h_shape)
        self.drop.build(h_shape)
        self.conv2.build(h_shape)

        self.res_proj.build(x_shape)
        self.skip_norm.build(x_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        x_shape, _ = input_shape
        b, h, w, _ = x_shape
        return (b, h, w, self.width)

    def call(
        self,
        inputs: tuple[Tensor, Tensor],
        *,
        skip_h: Tensor | None = None,
        training: bool | None = None,
        mask=None,
    ) -> Tensor:
        x, emb = inputs

        # Residual path
        assert self.res_proj is not None
        residual = self.res_proj(x, training=training)

        # Main path
        h = self.norm1(x, training=training)

        # Optional additive skip fusion (simple diffusion style)
        if self.skip_fuse != "none" and skip_h is not None:
            # This fuse assumes skip_h and x share the same channel dim.
            # (In U-ViT variants this is typically true.)
            sx = keras.ops.shape(x)[-1]
            ss = keras.ops.shape(skip_h)[-1]
            if sx is not None and ss is not None and sx != ss:
                raise ValueError(
                    f"skip_fuse='{self.skip_fuse}' requires skip_h and x to have the same channel dimension, "
                    f"but got x:C={sx} and skip_h:C={ss}."
                )
            s = self.skip_norm(skip_h, training=training)
            h = (h + s) / math.sqrt(2.0)

        h = self.act1(h)
        h = self.conv1(h, training=training)

        # Embedding injection (broadcast over H,W)
        e = self.emb_proj(emb, training=training)  # (B, 2*width) or (B,width)
        e = keras.ops.reshape(e, (-1, 1, 1, keras.ops.shape(e)[-1]))

        h = self.norm2(h, training=training)
        if self.use_scale_shift:
            scale, shift = keras.ops.split(e, 2, axis=-1)
            h = h * (1.0 + scale) + shift
        else:
            h = h + e

        h = self.act2(h)
        h = self.drop(h, training=training)
        h = self.conv2(h, training=training)

        return residual + h
