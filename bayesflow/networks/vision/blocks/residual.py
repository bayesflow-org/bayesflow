import math
from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from bayesflow.networks.vision.blocks.norms import SimpleNorm


@serializable("bayesflow.networks")
class ResidualBlock2D(keras.Layer):
    """
    Time-conditioned residual block for image backbones (NHWC).

    The block conditions on a *global* embedding vector (e.g., time / log-SNR embedding / sinusoidal embedding).
    Spatial conditioning maps (H, W, Cc) are intentionally not handled here; the backbone
    wrapper should decide where/how often to inject spatial conditions.

    Optionally supports "simple diffusion"-style additive skip fusion:
        h = (Norm(x) + Norm(skip_h)) / sqrt(2)
    controlled via `skip_fuse="add_sqrt2"` (see [1] for details.)

    [1] Hoogeboom et al. (2023), simple diffusion: End-to-end diffusion for high-resolution images
    """

    def __init__(
        self,
        width: int,
        *,
        activation: str = "swish",
        norm: Literal["layer", "group"] = "group",
        groups: int | None = 8,
        dropout: Literal[0, None] | float = 0.0,
        kernel_initializer: str | keras.initializers.Initializer = "he_normal",
        use_film: bool = True,
        skip_fuse_case: Literal["add_sqrt2"] | None = None,
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
        groups : int or None, optional
            Number of groups for group normalization. Adjusted in `build()` if needed
            to divide the channel dimension. Default is 8.
        dropout : float, optional
            Dropout rate applied before the second convolution. Default is 0.0.
        kernel_initializer : str or keras.initializers.Initializer, optional
            Initializer for conv1 and embedding projection. conv2 uses zero init to start
            near-identity (common in diffusion U-Nets). Default is "he_normal".
        use_film : bool, optional
            If True, uses FiLM-style scale-and-shift from t_emb.
            If False, uses additive embedding. Default is True.
        skip_fuse_case : {"add_sqrt2"} or None, optional
            If "add_sqrt2" and `skip_h` is passed at call time, fuses `x` and `skip_h` as
            (Norm(x) + Norm(skip_h)) / sqrt(2) before conv1. Default is "add_sqrt2".
        **kwargs
            Additional keyword arguments passed to `keras.Layer`.
        """
        super().__init__(**layer_kwargs(kwargs))
        self.width = int(width)
        self.activation = str(activation)
        self.norm = str(norm)
        self.groups = groups
        self.dropout = 0.0 if dropout is None else float(dropout)
        self.kernel_initializer = kernel_initializer
        self.use_film = bool(use_film)
        self.skip_fuse_case = skip_fuse_case

        match self.skip_fuse_case:
            # Optional skip normalization (only used when skip_h is passed)
            case "add_sqrt2":
                self.skip_fuse = self.skip_add_sqrt2
                self.skip_norm = SimpleNorm(
                    method=self.norm,
                    groups=self.groups,
                    center=True,
                    scale=True,
                    name="skip_norm",
                )
            case _:
                self.skip_fuse = None
                self.skip_norm = None

        # --- layers ---
        self.norm1 = SimpleNorm(
            method=self.norm,
            groups=self.groups,
            center=True,
            scale=True,
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

        emb_out = 2 * self.width if self.use_film else self.width
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

        # residual projection if channels mismatch (created in build)
        self.res_proj = None

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
            "use_film": self.use_film,
            "skip_fuse_case": self.skip_fuse_case,
        }
        return base | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return
        if len(input_shape) == 2:
            x_shape, emb_shape = input_shape
            skip_shape = None
        else:
            x_shape, emb_shape, skip_shape = input_shape
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
        self.res_proj.build(x_shape)
        self.emb_proj.build(emb_shape)

        # Build sublayers
        self.norm1.build(x_shape)
        self.act1.build(x_shape)
        self.conv1.build(x_shape)
        h_shape = self.conv1.compute_output_shape(x_shape)

        self.norm2.build(h_shape)
        self.act2.build(h_shape)
        self.drop.build(h_shape)
        self.conv2.build(h_shape)

        if self.skip_norm is not None:
            if skip_shape is None:
                raise ValueError(
                    f"skip_fuse_case='{self.skip_fuse_case}' requires skip_h input, but skip_h shape is {skip_shape}."
                )
            self.skip_norm.build(x_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        if len(input_shape) == 2:
            x_shape, _ = input_shape
        else:
            x_shape, _, _ = input_shape
        return tuple(x_shape[:-1]) + (self.width,)

    def skip_add_sqrt2(self, x: Tensor, skip_h: Tensor, training: bool | None = None) -> Tensor:
        # This fuse assumes skip_h and x share the same channel dim.
        # (In U-ViT variants this is typically true.)
        sx = x.shape[-1]
        ss = skip_h.shape[-1]
        if (sx is not None) and (ss is not None) and (sx != ss):
            raise ValueError(
                f"skip_fuse_case='{self.skip_fuse_case}' requires skip_h and x to have the same channel dimension, "
                f"but got x:C={sx} and skip_h:C={ss}."
            )
        s = self.skip_norm(skip_h, training=training)
        x = (x + s) / math.sqrt(2.0)
        return x

    def call(
        self,
        inputs: tuple[Tensor, Tensor, Tensor] | tuple[Tensor, Tensor],
        training: bool | None = None,
        **kwargs
    ) -> Tensor:
        if len(inputs) == 2:
            x, emb = inputs
            skip_h = None
        else:
            x, emb, skip_h = inputs

        # Residual path
        residual = self.res_proj(x, training=training)

        # Main path
        h = self.norm1(x, training=training)

        # Optional additive skip fusion (simple diffusion style)
        if self.skip_fuse is not None and skip_h is not None:
            h = self.skip_fuse(h, skip_h, training=training)

        h = self.act1(h)
        h = self.conv1(h, training=training)

        # Embedding injection (broadcast over H,W)
        e = self.emb_proj(emb, training=training)  # (B, 2*width) or (B,width)
        e = keras.ops.reshape(e, (-1, 1, 1, keras.ops.shape(e)[-1]))

        h = self.norm2(h, training=training)
        if self.use_film:
            scale, shift = keras.ops.split(e, 2, axis=-1)
            h = h * (1.0 + scale) + shift
        else:
            h = h + e

        h = self.act2(h)
        h = self.drop(h, training=training)
        h = self.conv2(h, training=training)

        return residual + h
