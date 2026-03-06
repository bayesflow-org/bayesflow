from collections.abc import Sequence
from typing import Literal

import keras

from bayesflow.utils import model_kwargs, logging
from bayesflow.utils.serialization import deserialize, serializable, serialize

from bayesflow.networks.residual import Residual
from .double_conv import DoubleConv
from ...networks import SummaryNetwork
from ...networks.transformers.attention import PoolingByMultiHeadAttention


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class ResNet(SummaryNetwork):
    """
    Implements a ResNet architecture. For more details see [1].

    [1] He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition.
    In Proceedings of the IEEE conference on computer vision and pattern recognition (pp. 770-778).
    arXiv:1512.03385
    """

    def __init__(
        self,
        summary_dim: int = 16,
        widths: Sequence[int] = (32, 64, 128),
        blocks_per_stage: int | Sequence[int] = 2,
        downsample_stage: Sequence[bool] | bool = True,
        norm: Literal["layer", "group", "batch"] | None = "layer",
        groups: int | None = None,
        dropout: float = 0.0,
        activation: str = "swish",
        down_mode: Literal["max_pool", "avg_pool", "conv"] = "avg_pool",
        pool_head: Literal["flatten", "global_avg", "global_max", "attention"] | keras.Layer = "global_avg",
        pool_num_heads: int = 4,
        hidden: int | None = None,
        **kwargs,
    ):
        super().__init__(**model_kwargs(kwargs))
        if norm != "batch" and activation in ("relu", "relu6", "leaky_relu"):
            logging.warning(
                f"Using ReLU-family activations with pre-activation ordering of {norm} norm "
                "can suppress negative inputs. Consider using 'swish' or 'mish', "
                "or set norm='batch' for post-activation ordering."
            )

        self.summary_dim = summary_dim
        self.widths = widths
        self.norm = norm
        self.groups = groups
        self.dropout = dropout
        self.activation = activation
        self.down_mode = down_mode
        self.pool_head = pool_head
        self.pool_num_heads = pool_num_heads

        self.blocks_per_stage = [blocks_per_stage] * len(widths) if isinstance(blocks_per_stage, int) else blocks_per_stage
        self.downsample_stage = [downsample_stage] * len(widths) if isinstance(downsample_stage, bool) else downsample_stage
        self.hidden = hidden or widths[-1]

        self.res_layers = []

        for s, width in enumerate(self.widths):
            for b in range(self.blocks_per_stage[s]):
                layer = DoubleConv(width, self.norm, self.groups, self.dropout, self.activation, residual=True)
                layer = Residual(layer)
                self.res_layers.append(layer)

            if self.downsample_stage[s]:
                if self.down_mode == "max_pool":
                    pool = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))
                elif self.down_mode == "avg_pool":
                    pool = keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
                elif self.down_mode == "conv":
                    pool = keras.layers.Conv2D(width, kernel_size=2, strides=2, padding="same")
                else:
                    raise ValueError(f"Unsupported downsampling mode: {down_mode}")
                self.res_layers.append(pool)

        if self.pool_head == "flatten":
            self.res_layers.append(keras.layers.Flatten())
        elif self.pool_head == "global_avg":
            self.res_layers.append(keras.layers.GlobalAveragePooling2D())
        elif self.pool_head == "global_max":
            self.res_layers.append(keras.layers.GlobalMaxPooling2D())
        elif self.pool_head == "attention":
            self.res_layers.append(keras.layers.Reshape((-1, widths[-1])))
            self.res_layers.append(PoolingByMultiHeadAttention(
                num_seeds=1,
                embed_dim=widths[-1],
                num_heads=pool_num_heads,
                dropout=dropout,
            ))
        elif isinstance(self.pool_head, keras.Layer):
            self.res_layers.append(self.pool_head)
        else:
            raise ValueError(f"Unsupported pooling head: {self.pool_head}")

        self.res_layers.append(keras.layers.Dense(self.hidden, activation=activation))
        self.res_layers.append(keras.layers.Dense(self.summary_dim))

    def call(self, inputs, training=False, **kwargs):
        x = inputs
        for layer in self.res_layers:
            x = layer(x, training=training)
        return x

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()

        config = {
            "summary_dim": self.summary_dim,
            "widths": self.widths,
            "blocks_per_stage": self.blocks_per_stage,
            "norm": self.norm,
            "groups": self.groups,
            "dropout": self.dropout,
            "activation": self.activation,
            "down_mode": self.down_mode,
            "pool_head": self.pool_head,
            "pool_num_heads": self.pool_num_heads,
            "hidden": self.hidden,
        }

        return base_config | serialize(config)
