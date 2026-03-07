from typing import Sequence, Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs, concatenate_valid
from bayesflow.utils.serialization import deserialize, serializable, serialize
from bayesflow.utils import check_lengths_same

from .blocks.norms import SimpleNorm
from .blocks.residual import ResidualBlock2D
from .blocks.attention import SelfAttention2D
from .blocks.downsample import DownSample2D
from .blocks.transformer import TransformerBlock2D
from .blocks.upsample import UpSample2D

from .embeddings.dense_fourier import DenseFourier


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
        transformer_dropout: float = 0.2,
        transformer_width: int | None = 1024,
        num_heads: int = 4,
        time_emb_dim: int = 32,
        time_emb: keras.Layer | None = None,
        time_emb_include_identity: bool = True,
        time_emb_use_residual_mlp: bool = True,
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
        time_emb_include_identity : bool, optional
            Whether the time embedding includes the original time scalar concatenated to the Fourier features.
            Default is True.
        time_emb_use_residual_mlp : bool, optional
            Whether the time embedding uses a residual MLP instead of a simple MLP. Default is True.
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
            Additional keyword arguments.
        """
        super().__init__(**layer_kwargs(kwargs))

        self.widths = widths
        self.res_blocks = (res_blocks,) * len(self.widths) if isinstance(res_blocks, int) else res_blocks
        self.attn_stage = (False,) * len(self.widths) if attn_stage is None else attn_stage
        self.dropout = (dropout,) * len(self.widths) if isinstance(dropout, float) else dropout

        self.transformer_blocks = transformer_blocks
        self.transformer_dropout = transformer_dropout
        self.transformer_width = 4 * self.widths[-1] if transformer_width is None else transformer_width
        self.num_heads = num_heads

        self.time_emb_dim = time_emb_dim
        self.time_emb_include_identity = time_emb_include_identity
        self.time_emb_use_residual_mlp = time_emb_use_residual_mlp
        self.use_film = use_film

        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.groups = groups
        self.norm = norm

        self.down_mode = down_mode
        self.up_kernel_size = up_kernel_size
        self.up_conv_first = up_conv_first

        check_lengths_same(self.res_blocks, self.widths, self.attn_stage, self.dropout)

        if time_emb is None:
            if self.time_emb_dim == 1:
                self.time_emb = keras.layers.Identity()
            else:
                self.time_emb = DenseFourier(
                    emb_dim=self.time_emb_dim,
                    include_identity=self.time_emb_include_identity,
                    use_residual_mlp=self.time_emb_use_residual_mlp,
                    kernel_initializer=self.kernel_initializer,
                )
        else:
            self.time_emb = time_emb

        self.input_projector = keras.layers.Conv2D(
            filters=self.widths[0],
            kernel_size=3,
            padding="same",
            kernel_initializer=self.kernel_initializer,
        )

        self.down_stage_names = []
        self.downsamples = []
        self.paddings = []

        for si, ch in enumerate(self.widths):
            blocks = []
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
                    )
                )
                if self.attn_stage[si]:
                    blocks.append(
                        SelfAttention2D(
                            num_heads=self.num_heads,
                            groups=self.groups,
                            residual="norm",
                            kernel_initializer=self.kernel_initializer,
                        )
                    )

            stage_name = f"down_stage_{si}"
            setattr(self, stage_name, blocks)
            self.down_stage_names.append(stage_name)
            self.downsamples.append(
                DownSample2D(
                    width=self.widths[si + 1] if si < len(self.widths) - 1 else self.transformer_width,
                    mode=self.down_mode,
                )
            )

        self.pos_emb = None
        self.trans_blocks = []
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
                )
            )

        self.upsamples = []
        self.up_stage_names = []
        self.crops = []

        for ri, ch in enumerate(reversed(self.widths)):
            si = (len(self.widths) - 1) - ri
            self.upsamples.append(
                UpSample2D(
                    width=self.widths[si],
                    kernel_size=self.up_kernel_size,
                    conv_first=self.up_conv_first,
                )
            )
            blocks = []
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
                    )
                )
                if self.attn_stage[si]:
                    blocks.append(
                        SelfAttention2D(
                            num_heads=self.num_heads,
                            groups=self.groups,
                            residual="norm",
                            kernel_initializer=self.kernel_initializer,
                        )
                    )
            stage_name = f"up_stage_{ri}"
            setattr(self, stage_name, blocks)
            self.up_stage_names.append(stage_name)

        self.out_norm = SimpleNorm(method=self.norm, groups=self.groups, center=True, scale=True)
        self.out_conv = None

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "widths": self.widths,
            "res_blocks": self.res_blocks,
            "transformer_blocks": self.transformer_blocks,
            "transformer_dropout": self.transformer_dropout,
            "transformer_width": self.transformer_width,
            "num_heads": self.num_heads,
            "time_emb_dim": self.time_emb_dim,
            "time_emb": self.time_emb,
            "time_emb_include_identity": self.time_emb_include_identity,
            "time_emb_use_residual_mlp": self.time_emb_use_residual_mlp,
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
        return base_config | serialize(config)

    def build(self, input_shape):
        if self.built:
            return

        assert len(input_shape) == 3, "UViT expects input shape to be a tuple of (x_shape, t_shape, cond_shape)"
        x_shape, t_shape, cond_shape = input_shape
        assert x_shape[-1] is not None, "UViT requires a known channel dimension for x."
        assert x_shape[1] is not None and x_shape[2] is not None, "UViT requires known spatial dimensions for x."

        t_shape = (t_shape[0], 1)
        self.time_emb.build(t_shape)
        t_emb_shape = self.time_emb.compute_output_shape(t_shape)

        # concatenate condition at beginning
        h_shape = list(x_shape)
        h_shape[-1] = x_shape[-1] + cond_shape[-1]
        h_shape = tuple(h_shape)
        self.input_projector.build(h_shape)
        h_shape = self.input_projector.compute_output_shape(h_shape)

        # down
        skip_shapes = []
        padding = []

        for si, stage_name in enumerate(self.down_stage_names):
            blocks = getattr(self, stage_name)
            for layer in blocks:
                if isinstance(layer, ResidualBlock2D):
                    layer.build((h_shape, t_emb_shape))
                    h_shape = layer.compute_output_shape((h_shape, t_emb_shape))
                    if self.attn_stage[si]:
                        continue
                else:
                    layer.build(h_shape)
                skip_shapes.append(h_shape)

            # Downsample and pad
            pad_h = h_shape[1] % 2 != 0
            pad_w = h_shape[2] % 2 != 0
            padding.append((pad_h, pad_w))
            layer = keras.layers.ZeroPadding2D(padding=((0, int(pad_h)), (0, int(pad_w))))
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
        )
        for ti, layer in enumerate(self.trans_blocks):
            layer.build((h_shape, t_emb_shape))
            h_shape = layer.compute_output_shape((h_shape, t_emb_shape))

        # up
        # for ri, blocks in enumerate(self.up_stages):
        for ri, stage_name in enumerate(self.up_stage_names):
            blocks = getattr(self, stage_name)
            si = (len(self.widths) - 1) - ri
            # Upsampling and Crop
            self.upsamples[ri].build(h_shape)
            h_shape = self.upsamples[ri].compute_output_shape(h_shape)
            pad_h, pad_w = padding[si]
            layer = keras.layers.Cropping2D(((0, int(pad_h)), (0, int(pad_w))))
            layer.build(h_shape)
            h_shape = layer.compute_output_shape(h_shape)
            self.crops.append(layer)
            for layer in blocks:
                if isinstance(layer, ResidualBlock2D):
                    skip_shape = skip_shapes.pop()
                    layer.build((h_shape, t_emb_shape, skip_shape))
                    h_shape = layer.compute_output_shape((h_shape, t_emb_shape, skip_shape))
                else:
                    layer.build(h_shape)

        self.out_norm.build(h_shape)
        self.out_conv = keras.layers.Conv2D(
            filters=int(x_shape[-1]),
            kernel_size=3,
            padding="same",
            kernel_initializer="zeros",
        )
        self.out_conv.build(h_shape)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0])

    def call(self, inputs: tuple[Tensor, Tensor, Tensor], training: bool = False) -> Tensor:
        x, t, cond = inputs
        x = self._prepare_inputs(x, cond)
        t_emb = self._compute_time_embedding(t, training=training)

        x, skips = self.encode(x, t_emb, training=training)
        x = self.bottleneck(x, t_emb, training=training)
        x = self.decode(x, t_emb, skips, training=training)

        x = self._project_output(x, training=training)
        return x

    def encode(self, x: Tensor, t_emb: Tensor, training: bool) -> tuple[Tensor, list[Tensor]]:
        skips = []

        for idx in range(len(self.down_stage_names)):
            x, skips = self._run_down_stage(idx, x, t_emb, skips, training=training)
            x = self.paddings[idx](x, training=training)
            x = self.downsamples[idx](x, training=training)

        return x, skips

    def bottleneck(self, x: Tensor, t_emb: Tensor, training: bool) -> Tensor:
        x = x + self.pos_emb
        for layer in self.trans_blocks:
            x = layer((x, t_emb), training=training)
        return x

    def decode(self, x: Tensor, t_emb: Tensor, skips: list[Tensor], training: bool) -> Tensor:
        for idx in range(len(self.up_stage_names)):
            x = self.upsamples[idx](x, training=training)
            x = self.crops[idx](x, training=training)
            x = self._run_up_stage(idx, x, t_emb, skips, training=training)

        return x

    def _prepare_inputs(self, x: Tensor, cond: Tensor) -> Tensor:
        x = concatenate_valid([x, cond], axis=-1)
        return self.input_projector(x)

    def _compute_time_embedding(self, t: Tensor, training: bool) -> Tensor:
        # Ensure shape [B, 1] even if t comes in with extra dims.
        t = keras.ops.reshape(t, (keras.ops.shape(t)[0], -1))[:, :1]
        return self.time_emb(t, training=training)

    def _run_down_stage(
        self, idx: int, x: Tensor, t_emb: Tensor, skips: list[Tensor], training: bool
    ) -> tuple[Tensor, list[Tensor]]:
        for layer in getattr(self, self.down_stage_names[idx]):
            is_residual = isinstance(layer, ResidualBlock2D)

            x = layer((x, t_emb), training=training) if is_residual else layer(x, training=training)

            # Don't store the residual output because the next layer is attention.
            if not (is_residual and self.attn_stage[idx]):
                skips.append(x)

        return x, skips

    def _run_up_stage(self, idx: int, x: Tensor, t_emb: Tensor, skips: list[Tensor], training: bool) -> Tensor:
        for layer in getattr(self, self.up_stage_names[idx]):
            if isinstance(layer, ResidualBlock2D):
                skip = skips.pop()
                x = layer((x, t_emb, skip), training=training)
            else:
                x = layer(x, training=training)
        return x

    def _project_output(self, x: Tensor, training: bool) -> Tensor:
        x = self.out_norm(x, training=training)
        return self.out_conv(x)
