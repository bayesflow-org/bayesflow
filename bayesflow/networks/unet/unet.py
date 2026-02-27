from typing import Sequence, Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs, concatenate_valid
from bayesflow.utils.serialization import deserialize, serializable, serialize
from bayesflow.utils import check_lengths_same

from .blocks.norms import SimpleNorm
from .blocks.residual import ResidualBlock2D
from .blocks.upsample import UpSample2D
from .blocks.downsample import DownSample2D
from .blocks.attention import SelfAttention2D

from .embeddings.dense_fourier import DenseFourier


@serializable("bayesflow.networks")
class UNet(keras.Layer):
    """
    Time-conditioned U-Net backbone for diffusion models [1].

    Expects inputs `(x, t, cond)`, where `cond` is concatenated channel-wise to `x` and a learned time embedding
    conditions residual blocks (optionally via FiLM). The network follows a DDPM-style encoder–decoder with skip
    connections, optional self-attention per stage, and pad/crop logic to support odd spatial sizes (see [1]).

    [1] Nain (2022) Keras example: Denoising Diffusion Probabilistic Model (https://keras.io/examples/generative/ddpm/)
    """

    def __init__(
        self,
        widths: Sequence[int] = (64, 128, 256, 512),
        res_blocks: Sequence[int] | int = 2,
        attn_stage: Sequence[bool] | None = (False, False, True, True),
        time_emb_dim: int = 32,
        time_emb: keras.Layer | None = None,
        use_film: bool = False,
        activation: str = "swish",
        kernel_initializer: str | keras.initializers.Initializer = "he_normal",
        dropout: Sequence[float] | float = 0.0,
        groups: int = 8,
        num_heads: int = 1,
        down_mode: Literal["average", "conv"] = "conv",
        up_kernel_size: Literal[1, 3] = 3,
        up_conv_first: bool = False,
        norm: Literal["layer", "group"] = "group",
        **kwargs,
    ):
        """
        Time-conditioned U-Net backbone for diffusion models.

        Parameters
        ----------
        widths : Sequence[int], optional
            Channel widths per resolution stage.
        res_blocks : Sequence[int] or int, optional
            Number of residual blocks per stage (decoder uses `+1` per stage).
        attn_stage : Sequence[bool] or None, optional
            Whether to use self-attention within each stage.
        time_emb_dim : int, optional
            Dimensionality of the time embedding. If 1, time is used directly.
        time_emb : keras.layers.Layer or None, optional
            Custom global time embedding layer. If None, uses `DenseFourier` when `time_emb_dim > 1`.
        use_film : bool, optional
            Whether residual blocks use FiLM or additive conditioning with the local time embedding.
        activation : str, optional
            Activation used throughout the network.
        kernel_initializer : str or keras.initializers.Initializer, optional
            Kernel initializer for convolution layers.
        dropout : Sequence[float] or float, optional
            Dropout rate used inside residual blocks. Default is 0.0.
        groups : int, optional
            Number of groups for group normalization where applicable.
        num_heads : int, optional
            Number of attention heads for self-attention layers. Default is 1.
        down_mode : {"average", "conv"}, optional
            "conv" uses a strided convolution, while "average" uses average pooling followed by a convolution.
             Default is "conv".
        up_kernel_size : {1, 3}, optional
            Kernel size for upsampling convolutions. Default is 3.
        up_conv_first : bool, optional
            If True, applies convolution before upsampling, after upsampling otherwise. Default is False.
        norm: Literal["layer", "group"], optional
            The type of normalization layer applied, defaults to "group"
        **kwargs
            Additional keyword arguments.
        """
        super().__init__(**layer_kwargs(kwargs))

        self.widths = widths
        self.res_blocks = (res_blocks,) * len(self.widths) if isinstance(res_blocks, int) else res_blocks
        self.attn_stage = (False,) * len(self.widths) if attn_stage is None else attn_stage

        self.time_emb_dim = time_emb_dim
        self.use_film = use_film
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.dropout = (dropout,) * len(self.widths) if isinstance(dropout, float) else dropout

        self.groups = groups
        self.num_heads = num_heads
        self.down_mode = down_mode
        self.up_kernel_size = up_kernel_size
        self.up_conv_first = up_conv_first

        self.norm = norm

        check_lengths_same(self.res_blocks, self.widths, self.attn_stage)
        check_lengths_same(self.dropout, self.widths)

        if time_emb is None:
            if self.time_emb_dim == 1:
                self.time_emb = keras.layers.Identity()
            else:
                self.time_emb = DenseFourier(
                    emb_dim=self.time_emb_dim,
                    include_identity=True,
                    use_residual_mlp=True,
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

        # Down pathway
        self.down_stages = []
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

            self.down_stages.append(blocks)

            if si < len(self.widths) - 1:
                self.downsamples.append(DownSample2D(width=self.widths[si + 1], mode=self.down_mode))

        self.mid1 = ResidualBlock2D(
            width=self.widths[-1],
            activation=self.activation,
            norm=self.norm,
            groups=self.groups,
            dropout=self.dropout[-1],
            kernel_initializer=self.kernel_initializer,
            use_film=self.use_film,
        )
        self.mid_attn = SelfAttention2D(
            num_heads=self.num_heads,
            groups=self.groups,
            residual="norm",
            kernel_initializer=self.kernel_initializer,
        )
        self.mid2 = ResidualBlock2D(
            width=self.widths[-1],
            activation=self.activation,
            norm=self.norm,
            groups=self.groups,
            dropout=self.dropout[-1],
            kernel_initializer=self.kernel_initializer,
            use_film=self.use_film,
        )

        self.upsamples = []
        self.up_stages = []
        self.crops = []

        for ri, ch in enumerate(reversed(self.widths)):
            si = (len(self.widths) - 1) - ri
            blocks = []
            for bi in range(self.res_blocks[si] + 1):
                blocks.append(
                    ResidualBlock2D(
                        width=ch,
                        activation=self.activation,
                        norm=self.norm,
                        groups=self.groups,
                        dropout=self.dropout[si],
                        kernel_initializer=self.kernel_initializer,
                        use_film=self.use_film,
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

            self.up_stages.append(blocks)

            if ri != len(self.widths) - 1:
                self.upsamples.append(
                    UpSample2D(
                        width=self.widths[si - 1],
                        kernel_size=self.up_kernel_size,
                        conv_first=self.up_conv_first,
                    )
                )

        self.out_norm = SimpleNorm(method=self.norm, groups=self.groups, center=True, scale=True)
        self.out_conv = None

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = layer_kwargs(super().get_config())
        cfg = {
            "widths": self.widths,
            "res_blocks": self.res_blocks,
            "attn_stage": self.attn_stage,
            "time_emb_dim": self.time_emb_dim,
            "time_emb": self.time_emb,
            "use_film": self.use_film,
            "activation": self.activation,
            "kernel_initializer": self.kernel_initializer,
            "dropout": self.dropout,
            "groups": self.groups,
            "num_heads": self.num_heads,
            "down_mode": self.down_mode,
            "up_kernel_size": self.up_kernel_size,
            "up_conv_first": self.up_conv_first,
            "norm": self.norm,
        }
        return base_config | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return

        assert len(input_shape) == 3, "UNet expects input shape to be a tuple of (x_shape, t_shape, cond_shape)"
        x_shape, t_shape, cond_shape = input_shape
        assert x_shape[-1] is not None, "UNet requires a known channel dimension for x."
        assert x_shape[1] is not None and x_shape[2] is not None, "UNet requires known spatial dimensions for x."

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
        skip_shapes = [h_shape]
        padding = []
        for si, blocks in enumerate(self.down_stages):
            for layer in blocks:
                if isinstance(layer, ResidualBlock2D):
                    layer.build((h_shape, t_emb_shape))
                    h_shape = layer.compute_output_shape((h_shape, t_emb_shape))
                    if self.attn_stage[si]:
                        continue
                else:  # self-attention
                    layer.build(h_shape)
                skip_shapes.append(h_shape)
            if si < len(self.widths) - 1:
                pad_h = h_shape[1] % 2 != 0
                pad_w = h_shape[2] % 2 != 0
                padding.append((pad_h, pad_w))
                layer = keras.layers.ZeroPadding2D(padding=((0, int(pad_h)), (0, int(pad_w))), name=f"down_s{si}_pad")
                layer.build(h_shape)
                h_shape = layer.compute_output_shape(h_shape)
                self.paddings.append(layer)
                self.downsamples[si].build(h_shape)
                h_shape = self.downsamples[si].compute_output_shape(h_shape)
                skip_shapes.append(h_shape)

        # mid
        self.mid1.build((h_shape, t_emb_shape))
        h_shape = self.mid1.compute_output_shape((h_shape, t_emb_shape))
        self.mid_attn.build(h_shape)
        self.mid2.build((h_shape, t_emb_shape))
        h_shape = self.mid2.compute_output_shape((h_shape, t_emb_shape))

        # up
        for ri, blocks in enumerate(self.up_stages):
            si = (len(self.widths) - 1) - ri
            for layer in blocks:
                if isinstance(layer, ResidualBlock2D):
                    skip_shape = skip_shapes.pop()
                    h_shape = list(h_shape)
                    h_shape[-1] = h_shape[-1] + skip_shape[-1]
                    h_shape = tuple(h_shape)
                    layer.build((h_shape, t_emb_shape))
                    h_shape = layer.compute_output_shape((h_shape, t_emb_shape))
                else:
                    layer.build(h_shape)
            if ri != len(self.widths) - 1:
                # Upsampling and Crop
                self.upsamples[ri].build(h_shape)
                h_shape = self.upsamples[ri].compute_output_shape(h_shape)
                pad_h, pad_w = padding[si - 1]
                layer = keras.layers.Cropping2D(((0, int(pad_h)), (0, int(pad_w))), name=f"up_s{si}_crop")
                layer.build(h_shape)
                h_shape = layer.compute_output_shape(h_shape)
                self.crops.append(layer)

        self.out_norm.build(h_shape)
        self.out_conv = keras.layers.Conv2D(
            filters=x_shape[-1],
            kernel_size=3,
            padding="same",
            kernel_initializer="zeros",
        )
        self.out_conv.build(h_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return tuple(input_shape[0])

    def call(self, inputs: tuple[Tensor, Tensor, Tensor], training: bool = False) -> Tensor:
        x, t, condition = inputs
        x = self._prepare_inputs(x, condition)
        t_emb = self._compute_time_embedding(t, training=training)

        x, skips = self.encode(x, t_emb, training=training)
        x = self.bottleneck(x, t_emb, training=training)
        x = self.decode(x, t_emb, skips, training=training)

        x = self._project_output(x, training=training)
        return x

    def encode(self, x: Tensor, t_emb: Tensor, training: bool) -> tuple[Tensor, list[Tensor]]:
        skips = [x]

        for idx, blocks in enumerate(self.down_stages):
            x, skips = self._run_down_stage(idx, x, t_emb, skips, training=training)

            if idx < len(self.downsamples):
                x = self.paddings[idx](x, training=training)
                x = self.downsamples[idx](x, training=training)
                skips.append(x)

        return x, skips

    def bottleneck(self, x: Tensor, t_emb: Tensor, training: bool) -> Tensor:
        x = self.mid1((x, t_emb), training=training)
        x = self.mid_attn(x, training=training)
        x = self.mid2((x, t_emb), training=training)
        return x

    def decode(self, x: Tensor, t_emb: Tensor, skips: list[Tensor], training: bool) -> Tensor:
        for idx, blocks in enumerate(self.up_stages):
            x = self._run_up_stage(idx, x, t_emb, skips, training=training)

            if idx != len(self.widths) - 1:
                x = self.upsamples[idx](x, training=training)
                x = self.crops[idx](x, training=training)

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
        for layer in self.down_stages[idx]:
            is_residual = isinstance(layer, ResidualBlock2D)

            x = layer((x, t_emb), training=training) if is_residual else layer(x, training=training)

            # Don't store the residual output because the next layer is attention.
            if not (is_residual and self.attn_stage[idx]):
                skips.append(x)

        return x, skips

    def _run_up_stage(self, idx: int, x: Tensor, t_emb: Tensor, skips: list[Tensor], training: bool) -> Tensor:
        for layer in self.up_stages[idx]:
            if isinstance(layer, ResidualBlock2D):
                skip = skips.pop()
                x = concatenate_valid([x, skip], axis=-1)
                x = layer((x, t_emb), training=training)
            else:
                x = layer(x, training=training)
        return x

    def _project_output(self, x: Tensor, training: bool) -> Tensor:
        x = self.out_norm(x, training=training)
        x = self.out_conv(x)
        return x
