from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize, deserialize
from bayesflow.utils import logging


@serializable("bayesflow.networks")
class DownSample2D(keras.Layer):
    """
    Implements a spatial downsampling layer for (B, H, W, C) tensors.

    This layer matches the common "vanilla DDPM U-Net" downsampling operation:
    a 3x3 convolution with stride 2 and "same" padding to reduce spatial resolution
    while projecting to `out_channels`. See https://keras.io/examples/generative/ddpm/

    The operation is:
        Conv2D(out_channels, kernel_size=3, strides=2, padding="same")

    """

    def __init__(
        self,
        out_channels: int,
        *,
        kernel_initializer: str | keras.Initializer = "he_normal",
        **kwargs,
    ):
        """
        Implements a spatial downsampling layer for (B, H, W, C) tensors.

        Parameters
        ----------
        out_channels : int
            Number of output channels after downsampling.
        kernel_initializer : str or keras.Initializer, optional
            Initialization strategy for convolution kernel weights. Default is "he_normal".
        **kwargs
            Additional keyword arguments passed to `keras.Layer`.
        """
        super().__init__(**layer_kwargs(kwargs))
        self.out_channels = int(out_channels)
        self.kernel_initializer = kernel_initializer

        self.down = keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=3,
            strides=2,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            name="down",
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = layer_kwargs(super().get_config())
        cfg = {
            "out_channels": self.out_channels,
            "kernel_initializer": self.kernel_initializer,
        }
        return base | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return
        # can be moved to unet since after adding pad and crop functions outside this warning won't be triggered here
        h, w = int(input_shape[1]), int(input_shape[2])
        if (h % 2 != 0) or (w % 2 != 0):
            logging.warning(
                f"{self.__class__.__name__}: received odd spatial dims (H={h}, W={w}). "
                "The U-Net wrapper will automatically pad/crop to keep skip merges shape-consistent. "
                "To avoid this behavior, provide inputs with H and W divisible by the total downsampling factor "
                "(typically 2**depth)."
            )
        self.down.build(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.down.compute_output_shape(input_shape)

    def call(self, inputs: Tensor, training: bool | None = None, mask=None) -> Tensor:
        return self.down(inputs, training=training)


@serializable("bayesflow.networks")
class UpSample2D(keras.Layer):
    """
    Implements a spatial upsampling layer for (B, H, W, C) tensors.

    This layer matches the common "vanilla DDPM U-Net" upsampling operation:
    nearest/bilinear resize by a factor of 2 followed by a 3x3 convolution.
    See https://keras.io/examples/generative/ddpm/.

    The operation is:
        UpSampling2D(size=2, interpolation=...) -> Conv2D(out_channels, kernel_size=3, padding="same")
    """

    def __init__(
        self,
        out_channels: int,
        *,
        interpolation: Literal["nearest", "bilinear"] = "nearest",
        kernel_initializer: str | keras.initializers.Initializer = "he_normal",
        **kwargs,
    ):
        """
        Implements a spatial upsampling layer for (B, H, W, C) tensors.

        Parameters
        ----------
        out_channels : int
            Number of output channels after upsampling.
        interpolation : {"nearest", "bilinear"}, optional
            Interpolation mode used by `UpSampling2D`. Default is "nearest".
        kernel_initializer : str or keras.Initializer, optional
            Initialization strategy for convolution kernel weights. Default is "he_normal".
        **kwargs
            Additional keyword arguments passed to `keras.Layer`.
        """
        super().__init__(**layer_kwargs(kwargs))
        self.out_channels = int(out_channels)
        self.interpolation = interpolation
        self.kernel_initializer = kernel_initializer

        self.up = keras.layers.UpSampling2D(size=2, interpolation=self.interpolation, name="up")
        self.conv = keras.layers.Conv2D(
            filters=self.out_channels,
            kernel_size=3,
            padding="same",
            kernel_initializer=self.kernel_initializer,
            name="conv",
        )

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = layer_kwargs(super().get_config())
        cfg = {
            "out_channels": self.out_channels,
            "interpolation": self.interpolation,
            "kernel_initializer": self.kernel_initializer,
        }
        return base | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return
        self.up.build(input_shape)
        up_shape = self.up.compute_output_shape(input_shape)
        self.conv.build(up_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        up_shape = self.up.compute_output_shape(input_shape)
        return self.conv.compute_output_shape(up_shape)

    def call(self, inputs: Tensor, training: bool | None = None, mask=None) -> Tensor:
        x = self.up(inputs, training=training)
        x = self.conv(x, training=training)
        return x
