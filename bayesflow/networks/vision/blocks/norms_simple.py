from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serialize, serializable, deserialize


@serializable("bayesflow.networks")
class SimpleNorm(keras.Layer):
    """
    Implements a lightweight normalization wrapper for vision backbones.

    Supports:
      - LayerNorm (method="layer")
      - GroupNorm (method="group")

    Notes
    -----
    For GroupNorm, `groups` must divide the number of channels along `axis`.
    If it does not, `groups` is reduced at build-time to the largest divisor
    <= the requested value.
    """
    def __init__(
        self,
        method: Literal["layer", "group"] = "group",
        *,
        groups: int = 8,
        axis: int = -1,
        center: bool = True,
        scale: bool = True,
        epsilon: float = 1e-3,
        **kwargs
    ):
        """
        Implements a lightweight normalization wrapper for vision backbones.

        Parameters
        ----------
        method : {"layer", "group"}, optional
            Type of normalization to apply. "layer" uses Layer Normalization; "group" uses Group Normalization.
            Default is "group".
        groups : int, optional
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
        epsilon : float, optional
            Small constant added for numerical stability. Diffusion-style architectures often use larger eps
            values than the Keras defaults. Default is 1e-3.
        **kwargs
            Additional keyword arguments passed to `keras.Layer`.
        """
        super().__init__(**layer_kwargs(kwargs))
        self.method = method
        self.groups = groups
        self.axis = axis
        self.center = center
        self.scale = scale
        self.epsilon = epsilon
        match method:
            case "layer":
                self.norm = keras.layers.LayerNormalization(
                    axis=axis,
                    center=center,
                    scale=scale,
                    epsilon=epsilon,
                )
            case "group":
                self.norm = keras.layers.GroupNormalization(
                    groups=groups,
                    axis=axis,
                    center=center,
                    scale=scale,
                    epsilon=epsilon,
                )
            case _:
                raise ValueError(f"Unsupported normalization method: {method}")

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base = super().get_config()
        base = layer_kwargs(base)
        cfg = {
            "method": self.method,
            "groups": self.groups,
            "axis": self.axis,
            "center": self.center,
            "scale": self.scale,
            "epsilon": self.epsilon,
        }
        return base | serialize(cfg)

    def build(self, input_shape):
        if self.built:
            return

        if self.method == "group":
            # infer channels along the normalization axis
            channels = input_shape[self.axis]
            if channels is not None:
                g = min(int(self.groups), int(channels))
                # find largest divisor <= requested groups
                while g > 1 and channels % g != 0:
                    g -= 1
                if channels % g != 0:
                    raise ValueError(
                        f"GroupNormalization: channels={channels} not divisible by any valid groups "
                        f"(requested groups={self.groups})."
                    )
                if g != self.groups:
                    # update stored value and recreate the layer
                    self.groups = g
                    self.norm = keras.layers.GroupNormalization(
                        groups=self.groups,
                        axis=self.axis,
                        center=self.center,
                        scale=self.scale,
                        epsilon=self.epsilon,
                    )
        self.norm.build(input_shape)

        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs: Tensor, training=None, **kwargs) -> Tensor:
        return self.norm(inputs, training=training)