import math

import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import find_network, layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.networks")
class Decoder(keras.Layer):
    """Decoder network that maps latent vectors back to the original space.

    Parameters
    ----------
    output_shape : int, tuple, or ``"auto"``, optional
        Per-sample shape of the reconstruction. ``"auto"`` (default) must be
        set by the parent before building.
    output_dim : int or ``"auto"``, optional
        Backward-compatible alias for a flat ``output_shape``. Ignored if
        ``output_shape`` is given explicitly.
    subnet : str, type, or keras.Layer, optional
        Feature subnet. Default ``"mlp"``.
    subnet_kwargs : dict, optional
        Extra arguments for the subnet.
    **kwargs
        Forwarded to ``keras.Layer``.
    """

    MLP_DEFAULT_CONFIG = {
        "widths": (128, 128),
        "activation": "mish",
        "kernel_initializer": "he_normal",
        "residual": True,
        "dropout": 0.0,
    }

    def __init__(
        self,
        output_shape: int | tuple | str = None,
        output_dim: int | str = "auto",
        subnet: str | type | keras.Layer = "mlp",
        subnet_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        if output_shape is None:
            output_shape = output_dim
        self._output_shape_arg = output_shape
        self._output_dim: int | None = None
        self.target_shape: tuple | None = None

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "mlp":
            subnet_kwargs = Decoder.MLP_DEFAULT_CONFIG | subnet_kwargs
        self.subnet = find_network(subnet, **subnet_kwargs)

        self._expand = None
        self._reshape = None
        self._flatten = None
        self.output_projector = None
        self._output_is_spatial = False

    @property
    def output_dim(self):
        return self._output_dim

    @output_dim.setter
    def output_dim(self, value):
        if not self.built:
            self._output_shape_arg = value
        self._output_dim = value if value is None or isinstance(value, int) else None

    def _resolve_target_shape(self) -> tuple:
        arg = self._output_shape_arg
        if arg == "auto" or arg is None:
            raise ValueError(
                "output_dim must be set before building the decoder (pass `output_shape` or `output_dim`)."
            )
        if isinstance(arg, int):
            return (arg,)
        return tuple(arg)

    def build(self, input_shape: Shape) -> None:
        if self.built:
            return

        self.target_shape = self._resolve_target_shape()
        self._output_is_spatial = len(self.target_shape) > 1
        self._output_dim = int(math.prod(self.target_shape))

        latent_is_flat = len(input_shape) == 2
        h_shape = input_shape

        # Bridge flat latent → spatial subnet input
        if latent_is_flat and self._output_is_spatial:
            flat = int(math.prod(self.target_shape))
            self._expand = keras.layers.Dense(units=flat, name="latent_expand")
            self._expand.build(h_shape)
            h_shape = self._expand.compute_output_shape(h_shape)
            self._reshape = keras.layers.Reshape(self.target_shape, name="latent_reshape")
            self._reshape.build(h_shape)
            h_shape = self._reshape.compute_output_shape(h_shape)

        self.subnet.build(h_shape)
        h_shape = self.subnet.compute_output_shape(h_shape)

        if self._output_is_spatial:
            expected = tuple(self.target_shape[:-1])
            actual = tuple(h_shape[1:-1])
            if actual != expected:
                raise ValueError(
                    f"Decoder subnet output spatial shape {actual} does not match "
                    f"the requested output spatial shape {expected}."
                )
            self.output_projector = keras.layers.Conv2D(
                filters=self.target_shape[-1], kernel_size=1, name="output_projector"
            )
        else:
            if len(h_shape) > 2:
                self._flatten = keras.layers.Flatten(name="decoder_flatten")
                self._flatten.build(h_shape)
                h_shape = self._flatten.compute_output_shape(h_shape)
            self.output_projector = keras.layers.Dense(units=self.target_shape[0], name="output_projector")

        self.output_projector.build(h_shape)

    def call(self, z: Tensor, training: bool = False) -> Tensor:
        """Decode latent ``z`` to shape ``(B, *target_shape)``."""
        h = z
        if self._expand is not None:
            h = self._expand(h, training=training)
            h = self._reshape(h)
        h = self.subnet(h, training=training)
        if self._flatten is not None:
            h = self._flatten(h)
        return self.output_projector(h, training=training)

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "output_shape": self._output_shape_arg,
            "subnet": self.subnet,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
