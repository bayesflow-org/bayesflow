from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize, deserialize


@serializable("bayesflow.networks")
class DownSample2D(keras.Layer):
    """
    Implements a spatial downsampling layer for (B, H, W, C) tensors.

    This layer matches can implement common downsampling schemes:
    average pooling with 1x1 convolution [1] or 3x3 convolution with stride 2 [2] to reduce spatial resolution
    while projecting to `width`.

    The operation is:
        .. code-block:: text

            Conv2D(width, kernel_size=3, strides=2, padding="same")
            or
            AveragePooling2D(pool_size=2, strides=2, padding="same") -> Conv2D(width, kernel_size=1, padding="same")

    [1] Hoogeboom et al. (2024) Simpler Diffusion (SiD2): 1.5 FID on ImageNet512 with pixel-space diffusion

    [2] Nain (2022) Keras example: Denoising Diffusion Probabilistic Model (https://keras.io/examples/generative/ddpm/)

    Parameters
    ----------
    width : int
        Number of output channels after downsampling.
    kernel_initializer : str or keras.Initializer, optional
        Initialization strategy for convolution kernel weights. Default is "he_normal".
    **kwargs
        Additional keyword arguments passed to `keras.Layer`.
    """

    def __init__(
        self,
        width: int,
        *,
        mode: Literal["average", "conv"] = "conv",
        kernel_initializer: str | keras.Initializer = "he_normal",
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        self.width = width
        self.kernel_initializer = kernel_initializer
        self.mode = mode

        match self.mode:
            case "conv":
                self.conv = keras.layers.Conv2D(
                    filters=self.width,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                )
            case "average":
                self.pool = keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="same")
                self.conv = keras.layers.Conv2D(
                    filters=self.width,
                    kernel_size=1,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                )
            case _:
                raise ValueError(f"Unsupported downsampling mode: {self.mode}")

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {"width": self.width, "mode": self.mode, "kernel_initializer": self.kernel_initializer}
        return base_config | serialize(config)

    def build(self, input_shape):
        if self.built:
            return
        if self.mode == "average":
            self.pool.build(input_shape)
            self.conv.build(self.pool.compute_output_shape(input_shape))
        else:
            self.conv.build(input_shape)

    def compute_output_shape(self, input_shape):
        if self.mode == "average":
            return self.conv.compute_output_shape(self.pool.compute_output_shape(input_shape))
        return self.conv.compute_output_shape(input_shape)

    def call(self, x: Tensor, **kwargs) -> Tensor:
        if self.mode == "average":
            return self.conv(self.pool(x))
        return self.conv(x)
