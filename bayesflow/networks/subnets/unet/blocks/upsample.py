from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize, deserialize


@serializable("bayesflow.networks")
class UpSample2D(keras.Layer):
    """
    Implements a spatial upsampling layer for (B, H, W, C) tensors.

    This layer matches the common "vanilla DDPM U-Net" upsampling operation:
    nearest/bilinear resize by a factor of 2 followed by a 1x1 [1] or 3x3 convolution [2].

    The operation is:
        .. code-block:: text

            UpSampling2D(size=2, interpolation=...) -> Conv2D(out_channels, kernel_size=..., padding="same")
            or
            Conv2D(out_channels, kernel_size=..., padding="same") -> UpSampling2D(size=2, interpolation=...)
    [1] Hoogeboom et al. (2023), simple diffusion: End-to-end diffusion for high-resolution images

    [2] Nain (2022) Keras example: Denoising Diffusion Probabilistic Model (https://keras.io/examples/generative/ddpm/)

    Parameters
    ----------
    width : int
        Number of output channels after upsampling.
    kernel_size : {1, 3}, optional
        Kernel size for the convolution applied before or after upsampling. Default is 1.
    conv_first : bool, optional
        If True, applies convolution before upsampling, after upsampling otherwise. Default is True.
    kernel_initializer : str or keras.Initializer, optional
        Initialization strategy for convolution kernel weights. Default is "he_normal".
    interpolation : {"nearest", "bilinear"}, optional
        Interpolation mode used by `UpSampling2D`. Default is "nearest".
    **kwargs
        Additional keyword arguments passed to `keras.Layer`.
    """

    def __init__(
        self,
        width: int,
        *,
        kernel_size: Literal[1, 3] = 1,
        conv_first: bool = True,
        kernel_initializer: str | keras.initializers.Initializer = "he_normal",
        interpolation: Literal["nearest", "bilinear"] = "nearest",
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        self.width = width
        self.kernel_size = kernel_size
        self.conv_first = conv_first
        self.interpolation = interpolation
        self.kernel_initializer = kernel_initializer

        self.conv = keras.layers.Conv2D(
            filters=self.width,
            kernel_size=self.kernel_size,
            padding="same",
            kernel_initializer=self.kernel_initializer,
        )
        self.upsample = keras.layers.UpSampling2D(size=2, interpolation=self.interpolation)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "width": self.width,
            "kernel_size": self.kernel_size,
            "conv_first": self.conv_first,
            "kernel_initializer": self.kernel_initializer,
            "interpolation": self.interpolation,
        }
        return base_config | serialize(config)

    def build(self, input_shape):
        if self.built:
            return
        if self.conv_first:
            self.conv.build(input_shape)
            self.upsample.build(self.conv.compute_output_shape(input_shape))
        else:
            self.upsample.build(input_shape)
            self.conv.build(self.upsample.compute_output_shape(input_shape))

    def compute_output_shape(self, input_shape):
        if self.conv_first:
            return self.upsample.compute_output_shape(self.conv.compute_output_shape(input_shape))
        return self.conv.compute_output_shape(self.upsample.compute_output_shape(input_shape))

    def call(self, x: Tensor, **kwargs) -> Tensor:
        if self.conv_first:
            return self.upsample(self.conv(x))
        return self.conv(self.upsample(x))
