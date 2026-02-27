from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize
from bayesflow.networks.vision.blocks.norms import SimpleNorm


@serializable("bayesflow.networks")
class TimeDense2D(keras.Layer):
    """
    Time-conditioned 2-layer Dense block for NHWC feature maps (B, H, W, C) from [1].

    Pattern:
        .. code-block:: text

            x_n = Norm(x)
            h   = Dense(width)(x_n)
            h   = inject(emb)      # FiLM or additive
            h   = act(h)
            h   = Dropout(h)
            h   = Dense(C, zero-init)(h)

    Returns according to `residual`:
        - "none": h
        - "input": x + h
        - "norm": x_n + h

    [1] Hoogeboom et al. (2023), simple diffusion: End-to-end diffusion for high-resolution images
    """

    def __init__(
        self,
        width: int,
        *,
        activation: str = "swish",
        norm: Literal["layer", "group"] = "layer",
        norm_with_bias: bool = False,
        groups: int | None = 8,
        dropout: float | None = 0.0,
        residual: Literal["none", "input", "norm"] = "none",
        kernel_initializer: str | keras.Initializer = "he_normal",
        use_film: bool = True,
        **kwargs,
    ):
        """
        Parameters
        ----------
        width : int
            Hidden width of the pointwise MLP (the intermediate channel size after `dense_up`).
        activation : str
            Activation after conditioning injection. Default "swish".
        norm : {"layer","group"}
            Pre-norm applied to `x` before the MLP. Default "layer".
        norm_with_bias : bool
            Whether the norm uses a bias/center term (e.g., LayerNorm/GroupNorm center). Default False.
        groups : int | None
            Number of groups if `norm="group"` (ignored for layer norm). Let `SimpleNorm` adjust/validate. Default 8.
        dropout : float | None
            Dropout rate applied after activation. Default 0.0 (None treated as 0.0).
        residual : {"none","input","norm"}
            Output mode:
              - "none": return delta
              - "input": return x + delta
              - "norm": return norm(x) + delta
            Default "none".
        kernel_initializer : str | keras.Initializer
            Initializer for `dense_up` and `emb_proj`. The output `dense_down` is always zero-init. Default "he_normal".
        use_film : bool
            If True, inject time via FiLM (scale+shift), requiring `emb_proj` to output `2*width`.
            If False, inject additively, requiring `emb_proj` to output `width`. Default True.
        **kwargs
            Additional keyword arguments passed to `keras.Layer`.
        """
        super().__init__(**layer_kwargs(kwargs))
        self.width = int(width)

        self.activation = str(activation)
        self.norm = str(norm)
        self.norm_with_bias = bool(norm_with_bias)
        self.groups = groups  # let SimpleNorm handle group adjustment / divisibility

        self.dropout = 0.0 if dropout is None else float(dropout)
        self.residual = str(residual)
        self.kernel_initializer = kernel_initializer

        self.use_film = bool(use_film)

        self.norm_layer = SimpleNorm(
            method=self.norm,
            groups=self.groups,
            axis=-1,
            center=self.norm_with_bias,
            scale=True,
            name="norm",
        )
        self.dense_up = keras.layers.Dense(
            self.width,
            kernel_initializer=self.kernel_initializer,
            name="dense_up",
        )
        emb_out = (2 * self.width) if self.use_film else self.width
        self.emb_proj = keras.layers.Dense(
            emb_out,
            kernel_initializer=self.kernel_initializer,
            name="emb_proj",
        )
        self.act = keras.layers.Activation(self.activation, name="act")
        self.drop = keras.layers.Dropout(self.dropout, name="dropout")

        # created/built in build()
        self.dense_down = None

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = layer_kwargs(super().get_config())
        cfg = {
            "width": self.width,
            "activation": self.activation,
            "norm": self.norm,
            "norm_with_bias": self.norm_with_bias,
            "groups": self.groups,
            "dropout": self.dropout,
            "residual": self.residual,
            "kernel_initializer": self.kernel_initializer,
            "use_film": self.use_film,
        }
        return base | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return

        # input_shape: (x_shape, emb_shape)
        x_shape, emb_shape = input_shape
        c_in = x_shape[-1]
        if c_in is None:
            raise ValueError("TimeDense2D requires a known channel dimension C in x_shape.")
        c_in = int(c_in)

        self.dense_down = keras.layers.Dense(
            c_in,
            kernel_initializer="zeros",
            name="dense_down_zero",
        )

        # build sublayers
        self.norm_layer.build(x_shape)
        self.dense_up.build(x_shape)
        h_shape = self.dense_up.compute_output_shape(x_shape)

        self.emb_proj.build(emb_shape)

        self.act.build(h_shape)
        self.drop.build(h_shape)
        self.dense_down.build(h_shape)

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

        x_n = self.norm_layer(x, training=training)
        h = self.dense_up(x_n, training=training)
        e = self.emb_proj(emb, training=training)  # (B, 2*width) or (B, width)
        e = keras.ops.reshape(e, (-1, 1, 1, keras.ops.shape(e)[-1]))  # broadcast over H,W
        if self.use_film:
            scale, shift = keras.ops.split(e, 2, axis=-1)
            h = h * (1.0 + scale) + shift
        else:
            h = h + e
        h = self.act(h)
        h = self.drop(h, training=training)
        h = self.dense_down(h, training=training)

        match self.residual:
            case "none":
                return h
            case "input":
                return x + h
            case "norm":
                return x_n + h
            case _:
                raise ValueError(f"Unknown residual mode: {self.residual!r}")
