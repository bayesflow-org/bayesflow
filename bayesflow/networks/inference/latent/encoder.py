import math

import keras
from keras import ops

from bayesflow.types import Shape, Tensor
from bayesflow.utils import find_network, layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.networks")
class Encoder(keras.Layer):
    """Encoder network that maps input to a latent distribution.
    
    Parameters
    ----------
    latent_shape : int, tuple, or ``"auto"``, optional
        Per-sample shape of the latent variable. ``int`` selects a flat latent;
        a ``tuple`` selects a spatial latent (e.g. ``(7, 7, 4)``). ``"auto"``
        (default) picks a flat latent of size ``max(2, prod(input_shape[1:]) // 2)``.
    latent_dim : int or ``"auto"``, optional
        Backward-compatible alias for a flat ``latent_shape``. Ignored if
        ``latent_shape`` is provided explicitly.
    subnet : str, type, or keras.Layer, optional
        Feature-extractor subnet. Default is ``"mlp"``.
    subnet_kwargs : dict, optional
        Extra arguments forwarded to the subnet constructor.
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
        latent_shape: int | tuple | str = None,
        latent_dim: int | str = "auto",
        subnet: str | type | keras.Layer = "mlp",
        subnet_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        if latent_shape is None:
            latent_shape = latent_dim
        self._latent_shape_arg = latent_shape

        self.latent_shape: tuple | None = None
        self.latent_dim: int | str = latent_dim if isinstance(latent_dim, str) else None
        self.seed_generator = keras.random.SeedGenerator()

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "mlp":
            subnet_kwargs = Encoder.MLP_DEFAULT_CONFIG | subnet_kwargs
        self.subnet = find_network(subnet, **subnet_kwargs)

        self._flatten = None
        self.mean_projector = None
        self.log_var_projector = None
        self._latent_is_spatial = False

    def _resolve_latent_shape(self, input_shape: Shape) -> tuple:
        arg = self._latent_shape_arg
        if arg == "auto":
            flat = 1
            for d in input_shape[1:]:
                flat *= int(d)
            return (max(2, flat // 2),)
        if isinstance(arg, int):
            return (arg,)
        return tuple(arg)

    def build(self, input_shape: Shape) -> None:
        if self.built:
            return

        self.latent_shape = self._resolve_latent_shape(input_shape)
        self._latent_is_spatial = len(self.latent_shape) > 1
        self.latent_dim = int(math.prod(self.latent_shape))

        self.subnet.build(input_shape)
        h_shape = self.subnet.compute_output_shape(input_shape)

        if self._latent_is_spatial:
            # Spatial latent: subnet must already produce matching spatial dims.
            expected = tuple(self.latent_shape[:-1])
            actual = tuple(h_shape[1:-1])
            if actual != expected:
                raise ValueError(
                    f"Encoder subnet output spatial shape {actual} does not match "
                    f"the requested latent spatial shape {expected}. Provide a subnet "
                    f"that downsamples the input to the target spatial dims."
                )
            self.mean_projector = keras.layers.Conv2D(
                filters=self.latent_shape[-1], kernel_size=1, name="mean_projector"
            )
            self.log_var_projector = keras.layers.Conv2D(
                filters=self.latent_shape[-1], kernel_size=1, name="log_var_projector"
            )
        else:
            if len(h_shape) > 2:
                self._flatten = keras.layers.Flatten(name="encoder_flatten")
                self._flatten.build(h_shape)
                h_shape = self._flatten.compute_output_shape(h_shape)
            self.mean_projector = keras.layers.Dense(units=self.latent_shape[0], name="mean_projector")
            self.log_var_projector = keras.layers.Dense(units=self.latent_shape[0], name="log_var_projector")

        self.mean_projector.build(h_shape)
        self.log_var_projector.build(h_shape)

    def call(self, x: Tensor, training: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """Encode ``x`` to a latent sample plus its distribution parameters.

        Returns ``(z, mean, log_var)``, each shaped ``(B, *latent_shape)``.
        """
        h = self.subnet(x, training=training)
        if self._flatten is not None:
            h = self._flatten(h)
        mean = self.mean_projector(h, training=training)
        log_var = self.log_var_projector(h, training=training)

        std = ops.exp(0.5 * log_var)
        epsilon = keras.random.normal(ops.shape(mean), seed=self.seed_generator)
        z = mean + std * epsilon
        return z, mean, log_var

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "latent_shape": self._latent_shape_arg,
            "subnet": self.subnet,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
