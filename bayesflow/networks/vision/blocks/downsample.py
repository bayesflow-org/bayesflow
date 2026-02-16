from typing import Literal

import keras

from bayesflow.networks import Sequential
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
    """

    def __init__(
        self,
        width: int,
        *,
        mode: Literal["average", "conv"] = "conv",
        kernel_initializer: str | keras.Initializer = "he_normal",
        **kwargs,
    ):
        """
        Implements a spatial downsampling layer for (B, H, W, C) tensors.

        Parameters
        ----------
        width : int
            Number of output channels after downsampling.
        kernel_initializer : str or keras.Initializer, optional
            Initialization strategy for convolution kernel weights. Default is "he_normal".
        **kwargs
            Additional keyword arguments passed to `keras.Layer`.
        """
        super().__init__(**layer_kwargs(kwargs))
        self.width = int(width)
        self.kernel_initializer = kernel_initializer
        self.mode = mode

        match self.mode:
            case "conv":
                self.down = keras.layers.Conv2D(
                    filters=self.width,
                    kernel_size=3,
                    strides=2,
                    padding="same",
                    kernel_initializer=self.kernel_initializer,
                    name="down",
                )
            case "average":
                self.down = Sequential([
                    keras.layers.AveragePooling2D(pool_size=2, strides=2, padding="same", name="avg_pool"),
                    keras.layers.Conv2D(
                        filters=self.width,
                        kernel_size=1,
                        padding="same",
                        kernel_initializer=self.kernel_initializer,
                        name="conv_proj",
                    ),
                ])
            case _:
                raise ValueError(f"Unsupported downsampling mode: {self.mode}")

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = layer_kwargs(super().get_config())
        cfg = {
            "width": self.width,
            "mode": self.mode,
            "kernel_initializer": self.kernel_initializer,
        }
        return base | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return
        self.down.build(input_shape)
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return self.down.compute_output_shape(input_shape)

    def call(self, inputs: Tensor, training: bool | None = None, mask=None) -> Tensor:
        return self.down(inputs, training=training)
