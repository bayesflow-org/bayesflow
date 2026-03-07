from collections.abc import Sequence
from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs, logging
from bayesflow.utils.serialization import serializable, serialize

from ...components.residual import Residual
from ...summary_network import SummaryNetwork
from ..transformers.attention import PoolingByMultiHeadAttention
from .double_conv import DoubleConv


@serializable("bayesflow.networks")
class ConvolutionalNetwork(SummaryNetwork):
    """A convolutional summary network with residual blocks.

    Uses a ResNet-style architecture [1]_ to compress 2D spatial inputs
    (e.g., images) into fixed-dimensional summary statistics.

    Each stage consists of one or more residual blocks (double convolution
    plus skip connection), optionally followed by spatial downsampling.  The
    final feature map is pooled and projected through a dense head.

    Parameters
    ----------
    summary_dim : int, optional
        Dimensionality of the output summary vector. Default is 16.
    widths : Sequence[int], optional
        Number of convolutional filters per stage. Default is ``(32, 64, 128)``.
    blocks_per_stage : int or Sequence[int], optional
        Residual blocks per stage. A single int is broadcast to every stage.
        Default is 2.
    downsample_stage : bool or Sequence[bool], optional
        Whether to spatially downsample after each stage. ``True`` is broadcast
        to every stage. Default is ``True``.
    norm : {"layer", "group", "batch"} or None, optional
        Normalization strategy inside residual blocks. Default is ``"layer"``.
    residual: bool, optional
        Whether to include skip connections around each double convolution block.
        Highly recommended for deeper networks. Default is ``True``.
    groups : int or None, optional
        Number of groups for group normalization. Default is ``None``.
    dropout : float, optional
        Dropout rate applied inside each residual block. Default is 0.0.
    activation : str, optional
        Activation function name. Default is ``"mish"``.
    down_mode : {"max_pool", "avg_pool", "conv"}, optional
        Spatial downsampling method. Default is ``"avg_pool"``.
    pool_head : {"flatten", "global_avg", "global_max", "attention"} or keras.Layer, optional
        Spatial-to-vector reduction before the dense head. Default is
        ``"global_avg"``.
    pool_num_heads : int, optional
        Number of attention heads when ``pool_head="attention"``. Default is 4.
    hidden : int or None, optional
        Width of the penultimate dense layer. Defaults to ``widths[-1]``.
    **kwargs
        Additional keyword arguments forwarded to :class:`SummaryNetwork`.

    References
    ----------
    .. [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual
       learning for image recognition. *Proceedings of the IEEE Conference on
       Computer Vision and Pattern Recognition*, 770-778. arXiv:1512.03385
    """

    def __init__(
        self,
        summary_dim: int = 16,
        widths: Sequence[int] = (32, 64, 128),
        blocks_per_stage: int | Sequence[int] = 2,
        downsample_stage: Sequence[bool] | bool = True,
        norm: Literal["layer", "group", "batch"] | None = "layer",
        residual: bool = True,
        groups: int | None = None,
        dropout: float = 0.0,
        activation: str = "mish",
        down_mode: Literal["max_pool", "avg_pool", "conv"] = "avg_pool",
        pool_head: Literal["flatten", "global_avg", "global_max", "attention"] | keras.Layer = "global_avg",
        pool_num_heads: int = 4,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        if norm != "batch" and activation in ("relu", "relu6", "leaky_relu"):
            logging.warning(
                f"Using ReLU-family activations with pre-activation ordering of {norm} norm "
                "can suppress negative inputs. Consider using 'swish' or 'mish', "
                "or set norm='batch' for post-activation ordering."
            )

        self.summary_dim = summary_dim
        self.widths = widths

        self.blocks_per_stage = (
            [blocks_per_stage] * len(widths) if isinstance(blocks_per_stage, int) else list(blocks_per_stage)
        )
        self.downsample_stage = (
            [downsample_stage] * len(widths) if isinstance(downsample_stage, bool) else list(downsample_stage)
        )

        self.norm = norm
        self.residual = residual
        self.groups = groups
        self.dropout = dropout
        self.activation = activation
        self.down_mode = down_mode
        self.pool_head = pool_head
        self.pool_num_heads = pool_num_heads
        self.layers = self._build_stages() + self._build_head()

    def _build_stages(self):
        layers = []
        for width, num_blocks, downsample in zip(self.widths, self.blocks_per_stage, self.downsample_stage):
            for _ in range(num_blocks):
                block = DoubleConv(width, self.norm, self.groups, self.dropout, self.activation)
                layers.append(Residual(block) if self.residual else block)

            if downsample:
                layers.extend(self._make_downsample_layers(width))

        return layers

    def _make_downsample_layers(self, width: int):
        # pad odd spatial dims to even so 2x2 pooling divides cleanly
        pad = keras.layers.Lambda(
            lambda x: keras.ops.pad(
                x,
                [[0, 0], [0, keras.ops.shape(x)[1] % 2], [0, keras.ops.shape(x)[2] % 2], [0, 0]],
            )
        )

        match self.down_mode:
            case "max_pool":
                pool = keras.layers.MaxPool2D(pool_size=2, strides=2)
            case "avg_pool":
                pool = keras.layers.AveragePooling2D(pool_size=2, strides=2)
            case "conv":
                pool = keras.layers.Conv2D(width, kernel_size=2, strides=2, padding="same")
            case _:
                raise ValueError(f"Unsupported downsampling mode: {self.down_mode!r}")

        return [pad, pool]

    def _build_head(self):
        layers = self._make_pool_layers()
        layers.append(keras.layers.Dense(self.summary_dim))
        return layers

    def _make_pool_layers(self):
        if isinstance(self.pool_head, keras.Layer):
            return [self.pool_head]
        match self.pool_head:
            case "flatten":
                return [keras.layers.Flatten()]
            case "global_avg":
                return [keras.layers.GlobalAveragePooling2D()]
            case "global_max":
                return [keras.layers.GlobalMaxPooling2D()]
            case "attention":
                return [
                    keras.layers.Reshape((-1, self.widths[-1])),
                    PoolingByMultiHeadAttention(
                        num_seeds=1, embed_dim=self.widths[-1], num_heads=self.pool_num_heads, dropout=self.dropout
                    ),
                ]
            case _:
                raise ValueError(f"Unsupported pooling head: {self.pool_head!r}")

    def call(self, x: Tensor, training: bool = False, **kwargs):
        for layer in self.layers:
            x = layer(x, training=training)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "summary_dim": self.summary_dim,
            "widths": self.widths,
            "blocks_per_stage": self.blocks_per_stage,
            "norm": self.norm,
            "residual": self.residual,
            "groups": self.groups,
            "dropout": self.dropout,
            "activation": self.activation,
            "down_mode": self.down_mode,
            "pool_head": self.pool_head,
            "pool_num_heads": self.pool_num_heads,
        }
        return base_config | serialize(config)
