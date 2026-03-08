from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs, logging
from bayesflow.utils.serialization import serialize, serializable, deserialize


@serializable("bayesflow.networks")
class SimpleNorm(keras.Layer):
    """
    Implements a lightweight normalization wrapper for vision backbones.

    Supports:
      - LayerNorm (method="layer")
      - GroupNorm (method="group")

    Parameters
    ----------
    method : {"layer", "group"}, optional
        Type of normalization to apply. "layer" uses Layer Normalization; "group" uses Group Normalization.
        Default is "group".
    groups : int or None, optional
        Number of groups for Group Normalization. Only used when `method="group"`. At build time, if the
        requested value does not divide the number of channels, it is reduced to the largest valid divisor
        <= `groups`. Default is 8.
    axis : int, optional
        Channel axis along which normalization is applied. For channels-last tensors (B,H,W,C), use -1.
        Default is -1.
    center : bool, optional
        Whether to include a learnable offset (beta). Default is True.
    scale : bool, optional
        Whether to include a learnable scale (gamma). Default is True.
    gamma_initializer : str, optional
        Initializer for the gamma weights when `scale=True`. Default is "ones".
    **kwargs
        Additional keyword arguments passed to `keras.Layer`.

    Notes
    -----
    For GroupNorm, `groups` must divide the number of channels along `axis`.
    If it does not, `groups` is reduced at build-time to the largest divisor
    <= the requested value.
    """

    def __init__(
        self,
        method: Literal["layer", "group", "batch"] | None = "group",
        *,
        groups: int | None = 8,
        axis: int = -1,
        center: bool = True,
        scale: bool = True,
        gamma_initializer: str = "ones",
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        self.method = method
        self.groups = groups
        self.axis = axis
        self.center = center
        self.scale = scale
        self.gamma_initializer = gamma_initializer

        match method:
            case "layer":
                self.norm = keras.layers.LayerNormalization(
                    axis=axis, center=center, scale=scale, gamma_initializer=gamma_initializer
                )
            case "group":
                self.norm = None
            case "batch":
                self.norm = keras.layers.BatchNormalization(
                    axis=axis, center=center, scale=scale, gamma_initializer=gamma_initializer
                )
            case None:
                self.norm = keras.layers.Identity()
            case _:
                raise ValueError(f"Unsupported normalization method: {method}")

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "method": self.method,
            "groups": self.groups,
            "axis": self.axis,
            "center": self.center,
            "scale": self.scale,
            "gamma_initializer": self.gamma_initializer,
        }
        return base_config | serialize(config)

    def build(self, input_shape):
        if self.built:
            return

        if self.method == "group":
            adjusted_groups = self.groups
            # infer channels along the normalization axis
            channels = input_shape[self.axis]
            if channels is not None:
                adjusted_groups = min(self.groups, channels)
                # find largest divisor <= requested groups
                while adjusted_groups > 1 and channels % adjusted_groups != 0:
                    adjusted_groups -= 1

                if adjusted_groups != self.groups:
                    # update stored value and recreate the layer
                    logging.warning(
                        f"Adjusted groups from {self.groups} to {adjusted_groups} to fit input channels {channels}."
                    )

            self.norm = keras.layers.GroupNormalization(
                groups=adjusted_groups, axis=self.axis, center=self.center, scale=self.scale
            )
        self.norm.build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs: Tensor, training=None, **kwargs) -> Tensor:
        return self.norm(inputs, training=training)
