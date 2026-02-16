from typing import Sequence, Literal

import keras

from bayesflow.networks.vision.blocks.transformer import TransformerBlock2D
from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs, concatenate_valid
from bayesflow.utils.serialization import deserialize, serializable, serialize
from bayesflow.utils import check_lengths_same

from bayesflow.networks.vision.blocks.norms import SimpleNorm
from bayesflow.networks.vision.blocks.residual import ResidualBlock2D
from bayesflow.networks.vision.blocks.upsample import UpSample2D
from bayesflow.networks.vision.blocks.downsample import DownSample2D
from bayesflow.networks.vision.blocks.attention import SelfAttention2D
from bayesflow.networks.vision.embeddings.dense_fourier import DenseFourier


@serializable("bayesflow.networks")
class UViT(keras.Layer):
    """
    Time-conditioned U-ViT backbone for diffusion models.

    Expects inputs `(x, t, cond)`, where `cond` is concatenated channel-wise to `x` and a learned time embedding
    conditions all residual and transformer blocks (optionally via FiLM). The network uses a DDPM-style convolutional
    encoder–decoder with skip connections at higher resolutions, but replaces the low-resolution bottleneck with a
    stack of transformer blocks operating on NHWC feature maps. A learned 2D positional embedding is added at the
    bottleneck resolution.

    Follows the UViT architecture from [1], with some modifications and generalizations:
      - Flexible widths and block counts per stage (not necessarily doubling channels at each downsample).
      - Optional self-attention blocks after each residual block within a stage.
      - Configurable downsampling mode ("conv" vs "average").
      - More flexible time embedding options (custom layer, FiLM vs additive, etc.).

    [1] Hoogeboom et al. (2023), simple diffusion: End-to-end diffusion for high-resolution images
    """
    def __init__(
        self,
        widths: Sequence[int] = (64, 128, 256),
        res_blocks: Sequence[int] | int = 2,
        transformer_blocks: int = 3,
        *,
        transformer_dropout: float = 0.2,
        transformer_width: int | None = 1024,
        num_heads: int = 4,
        time_emb_dim: int = 32,
        time_emb: keras.Layer | None = None,
        use_film: bool = True,
        activation: str = "swish",
        kernel_initializer: str | keras.initializers.Initializer = "he_normal",
        dropout: Sequence[float] | float = 0.0,
        norm: Literal["layer", "group"] = "group",
        groups: int = 8,
        attn_stage: Sequence[bool] | None = None,
        down_mode: Literal["conv", "average"] = "average",
        up_kernel_size: Literal[1, 3] = 1,
        up_conv_first: bool = True,
        **kwargs,
    ):
        """
        Time-conditioned U-ViT backbone for diffusion models.

        Parameters
        ----------
        widths : Sequence[int], optional
            Channel widths per convolutional resolution stage (encoder/decoder stages).
        res_blocks : Sequence[int] or int, optional
            Number of residual blocks per stage (same count in encoder and decoder).
        transformer_blocks : int, optional
            Number of transformer blocks applied at the bottleneck resolution.
        transformer_dropout : float, optional
            Dropout rate used inside the bottleneck transformer MLP sub-blocks.
        transformer_width : int or None, optional
            Channel width used at the bottleneck and within transformer blocks. If None, uses `widths[-1]`.
        num_heads : int, optional
            Number of attention heads for self-attention in both stage attention blocks and bottleneck transformers.
        time_emb_dim : int, optional
            Dimensionality of the time embedding. If 1, time is used directly.
        time_emb : keras.layers.Layer or None, optional
            Custom global time embedding layer. If None, uses `DenseFourier` when `time_emb_dim > 1`.
        use_film : bool, optional
            Whether embedding injection uses FiLM (scale/shift) or additive conditioning.
        activation : str, optional
            Activation used throughout the network.
        kernel_initializer : str or keras.initializers.Initializer, optional
            Kernel initializer used for convolution and dense layers (except explicitly zero-initialized projections).
        dropout : Sequence[float] or float, optional
            Dropout rate used inside residual blocks. Default is 0.0.
        norm : {"layer", "group"}, optional
            Normalization type used throughout the backbone.
        groups : int, optional
            Number of groups for group normalization where applicable.
        attn_stage : Sequence[bool] or None, optional
            Whether to insert an additional self-attention block after each residual block within a stage.
        down_mode : {"average", "conv"}, optional
            Downsampling mode. "conv" uses a strided convolution; "average" uses average pooling followed by a
            convolution/projection. Default is "average".
        up_kernel_size : {1, 3}, optional
            Kernel size for the convolution used in the upsampling block. Default is 1.
        up_conv_first : bool, optional
            If True, applies the convolution before upsampling in the upsampling block. Default is True.
        **kwargs
            Additional keyword arguments (e.g., `time_emb_include_identity`, `time_emb_use_residual_mlp`).

        Notes
        -----
        - Expected inputs in `call()` are a tuple ``(x, t, cond)``.
        - `x` is an NHWC tensor of shape ``(B, H, W, Cx)`` and defines the output channel dimension.
        - `cond` is expected to be broadcast-compatible for channel-wise concatenation with `x` (typically
          ``(B, H, W, Cc)``).
        - Downsampling is applied after every convolutional stage (including the last) to reach the transformer
          bottleneck resolution; the decoder upsamples symmetrically.
        - The model pads bottom/right before each downsampling step if H/W are odd, and crops after the corresponding
          upsampling step so the final spatial dimensions match the input.
        - Decoder residual blocks use skip fusion when a skip tensor is provided (e.g., simple-diffusion-style
          additive fusion via `skip_fuse_case="add_sqrt2"` in `ResidualBlock2D`).
        """
        super().__init__(**layer_kwargs(kwargs))

        self.widths = widths
        self.res_blocks = (res_blocks,) * len(self.widths) if isinstance(res_blocks, int) else res_blocks
        self.attn_stage = (False,) * len(self.widths) if attn_stage is None else attn_stage
        self.dropout = (float(dropout),) * len(self.widths) if isinstance(dropout, float) else dropout
        check_lengths_same(self.res_blocks, self.widths, self.attn_stage, self.dropout)

        self.transformer_blocks = int(transformer_blocks)
        self.transformer_dropout = float(transformer_dropout)
        self.transformer_width = int(4*self.widths[-1]) if transformer_width is None else int(transformer_width)
        self.time_emb_dim = int(time_emb_dim)
        self.use_film = bool(use_film)
        self.activation = str(activation)
        self.kernel_initializer = kernel_initializer
        self.groups = int(groups)
        self.norm = str(norm)
        self.num_heads = int(num_heads)
        self.down_mode = down_mode
        self.up_kernel_size = up_kernel_size
        self.up_conv_first = bool(up_conv_first)


        # --- Time embedding ---
        if time_emb is None:
            if self.time_emb_dim == 1:
                self.time_emb = keras.layers.Identity()
            else:
                self.time_emb = DenseFourier(
                    emb_dim=self.time_emb_dim,
                    include_identity=kwargs.get("time_emb_include_identity", True),
                    use_residual_mlp=kwargs.get("time_emb_use_residual_mlp", True),
                    kernel_initializer=self.kernel_initializer,
                    name="time_emb",
                )
        else:
            self.time_emb = time_emb

        # --- input projection ---
        self.proj_in = keras.layers.Conv2D(
            filters=self.widths[0],
            kernel_size=3,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            name="proj_in",
        )

        # --- down path ---
        self.down_stages: list[list[keras.Layer]] = []
        self.downsamples: list[keras.Layer] = []
        self.paddings: list[keras.Layer] = []
        for si, ch in enumerate(self.widths):
            blocks: list[keras.Layer] = []
            for bi in range(self.res_blocks[si]):
                blocks.append(
                    ResidualBlock2D(
                        width=ch,
                        activation=self.activation,
                        norm=self.norm,
                        groups=self.groups,
                        dropout=self.dropout[si],
                        kernel_initializer=self.kernel_initializer,
                        use_film=self.use_film,
                        skip_fuse_case=None,
                        name=f"down_s{si}_b{bi}",
                    )
                )
                if self.attn_stage[si]:
                    blocks.append(
                        SelfAttention2D(
                            num_heads=self.num_heads,
                            groups=self.groups,
                            residual="norm",
                            kernel_initializer=self.kernel_initializer,
                            name=f"down_s{si}_b{bi}_attn",
                        )
                    )

            self.down_stages.append(blocks)
            self.downsamples.append(
                DownSample2D(
                    width=self.widths[si+1] if si < len(self.widths) - 1 else self.transformer_width,
                    mode=self.down_mode,
                    name=f"down_s{si}_ds"
                )
            )

        # --- bottleneck ---
        self.pos_emb = None
        self.trans_blocks: list[keras.Layer] = []
        for i in range(self.transformer_blocks):
            self.trans_blocks.append(
                TransformerBlock2D(
                    width=self.transformer_width,
                    num_heads=self.num_heads,
                    attn_groups=self.groups,
                    mlp_groups=self.groups,
                    mlp_dropout=self.transformer_dropout,
                    mlp_use_film=self.use_film,
                    kernel_initializer=self.kernel_initializer,
                    name=f"mid_trans_{i}",
                )
            )

        # --- up path ---
        self.upsamples: list[keras.Layer] = []
        self.up_stages: list[list[keras.Layer]] = []
        self.crops: list[keras.Layer] = []
        # build decoder stages in reverse order
        for ri, ch in enumerate(reversed(self.widths)):
            si = (len(self.widths) - 1) - ri
            self.upsamples.append(
                UpSample2D(
                    width=self.widths[si],
                    kernel_size=self.up_kernel_size,
                    conv_first=self.up_conv_first,
                    name=f"up_s{si}_us"
                )
            )
            blocks: list[keras.Layer] = []
            for bi in range(self.res_blocks[si]):
                blocks.append(
                    ResidualBlock2D(
                        width=ch,
                        activation=self.activation,
                        norm=self.norm,
                        groups=self.groups,
                        dropout=self.dropout[si],
                        kernel_initializer=self.kernel_initializer,
                        use_film=self.use_film,
                        skip_fuse_case="add_sqrt2",
                        name=f"up_s{si}_b{bi}",
                    )
                )
                if self.attn_stage[si]:
                    blocks.append(
                        SelfAttention2D(
                            num_heads=self.num_heads,
                            groups=self.groups,
                            residual="norm",
                            kernel_initializer=self.kernel_initializer,
                            name=f"up_s{si}_b{bi}_attn",
                        )
                    )
            self.up_stages.append(blocks)


        # --- head ---
        self.out_norm = SimpleNorm(
            method=self.norm,
            groups=self.groups,
            center=True,
            scale=True,
            name="out_norm",
        )
        self.out_act = keras.layers.Activation(self.activation, name="out_act")
        self.out_conv = None

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = layer_kwargs(super().get_config())
        cfg = {
            "widths": self.widths,
            "res_blocks": self.res_blocks,
            "transformer_blocks": self.transformer_blocks,
            "transformer_dropout": self.transformer_dropout,
            "transformer_width": self.transformer_width,
            "num_heads": self.num_heads,
            "time_emb_dim": self.time_emb_dim,
            "time_emb": self.time_emb,
            "use_film": self.use_film,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "dropout": self.dropout,
            "norm": self.norm,
            "groups": self.groups,
            "attn_stage": self.attn_stage,
            "down_mode": self.down_mode,
            "up_kernel_size": self.up_kernel_size,
            "up_conv_first": self.up_conv_first,
        }
        return base | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return

        assert len(input_shape) == 3, "UViT expects input shape to be a tuple of (x_shape, t_shape, cond_shape)"
        x_shape, t_shape, cond_shape = input_shape
        assert x_shape[-1] is not None, "UViT requires a known channel dimension for x."
        assert x_shape[1] is not None and x_shape[2] is not None, "UViT requires known spatial dimensions for x."

        self.time_emb.build(t_shape)
        t_emb_shape = self.time_emb.compute_output_shape(t_shape)

        # concatenate condition at beginning
        h_shape = list(x_shape)
        h_shape[-1] = x_shape[-1] + cond_shape[-1]
        h_shape = tuple(h_shape)
        self.proj_in.build(h_shape)
        h_shape = self.proj_in.compute_output_shape(h_shape)

        # down
        skip_shapes = []
        padding = []
        for si, blocks in enumerate(self.down_stages):
            for layer in blocks:
                if isinstance(layer, ResidualBlock2D):
                    layer.build((h_shape, t_emb_shape))
                    h_shape = layer.compute_output_shape((h_shape, t_emb_shape))
                    if self.attn_stage[si]:
                        continue
                else: # self-attention
                    layer.build(h_shape)
                skip_shapes.append(h_shape)
            # Downsampling and Pad
            pad_h = (h_shape[1] % 2 != 0)
            pad_w = (h_shape[2] % 2 != 0)
            padding.append((pad_h, pad_w))
            layer = keras.layers.ZeroPadding2D(padding=((0, int(pad_h)), (0, int(pad_w))), name=f"down_s{si}_pad")
            layer.build(h_shape)
            h_shape = layer.compute_output_shape(h_shape)
            self.paddings.append(layer)
            self.downsamples[si].build(h_shape)
            h_shape = self.downsamples[si].compute_output_shape(h_shape)

        # mid
        self.pos_emb = self.add_weight(
            shape=(1,) + tuple(h_shape[1:]),
            initializer=keras.initializers.RandomNormal(stddev=0.01),
            trainable=True,
            name="pos_emb"
        )
        for ti, layer in enumerate(self.trans_blocks):
            layer.build((h_shape, t_emb_shape))
            h_shape = layer.compute_output_shape((h_shape, t_emb_shape))

        # up
        for ri, blocks in enumerate(self.up_stages):
            si = (len(self.widths) - 1) - ri

            # Upsampling and Crop
            self.upsamples[ri].build(h_shape)
            h_shape = self.upsamples[ri].compute_output_shape(h_shape)
            pad_h, pad_w = padding[si]
            layer = keras.layers.Cropping2D(((0, int(pad_h)), (0, int(pad_w))), name=f"up_s{si}_crop")
            layer.build(h_shape)
            h_shape = layer.compute_output_shape(h_shape)
            self.crops.append(layer)

            for layer in blocks:
                if isinstance(layer, ResidualBlock2D):
                    skip_shape = skip_shapes.pop()
                    layer.build((h_shape, t_emb_shape, skip_shape))
                    h_shape = layer.compute_output_shape((h_shape, t_emb_shape, skip_shape))
                else: # self-attention
                    layer.build(h_shape)


        self.out_norm.build(h_shape)
        self.out_act.build(h_shape)
        self.out_conv = keras.layers.Conv2D(
            filters=int(x_shape[-1]),
            kernel_size=3,
            padding="same",
            kernel_initializer="zeros",
            name="out_conv_zero",
        )
        self.out_conv.build(h_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0])

    def call(
        self,
        inputs: tuple[Tensor, Tensor, Tensor],
        training: bool | None = None,
        mask=None,
    ) -> Tensor:
        assert len(inputs) == 3, "UViT expects inputs to be a tuple of (x, t, cond)"
        x, t, cond = inputs
        assert cond is not None, "UViT currently requires a condition input."

        t_emb = self.time_emb(t, training=training)

        x = concatenate_valid([x, cond], axis=-1)
        x = self.proj_in(x, training=training)

        # encoder
        skips: list[Tensor] = []
        for si, blocks in enumerate(self.down_stages):
            for layer in blocks:
                if isinstance(layer, ResidualBlock2D):
                    x = layer((x, t_emb), training=training)
                    if self.attn_stage[si]:
                        continue
                else:
                    x = layer(x, training=training)
                skips.append(x)
            x = self.paddings[si](x, training=training)
            x = self.downsamples[si](x, training=training)

        # bottleneck
        x = x + self.pos_emb
        for layer in self.trans_blocks:
            x = layer((x, t_emb), training=training)

        # decoder (reverse skips)
        for ri, blocks in enumerate(self.up_stages):
            x = self.upsamples[ri](x, training=training)
            x = self.crops[ri](x, training=training)
            for layer in blocks:
                if isinstance(layer, ResidualBlock2D):
                    skip = skips.pop()
                    x = layer((x, t_emb, skip), training=training)
                else: # self-attention
                    x = layer(x, training=training)


        x = self.out_norm(x, training=training)
        x = self.out_act(x)
        x = self.out_conv(x, training=training)
        return x