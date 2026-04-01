import keras
from keras import ops

from bayesflow.types import Shape, Tensor
from bayesflow.utils import find_network, layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.networks")
class Encoder(keras.Layer):
    """Encoder network that maps input to a latent distribution.

    Maps input x to latent distribution parameters (mean, log_var) and
    samples z using the reparameterization trick: z = mean + std * epsilon.

    Can be used standalone or as part of a :py:class:`LatentInferenceNetwork`.
    When used standalone, the encoder can compress data into a latent space
    for use with any inference network (e.g., ``FlowMatching``, ``DiffusionModel``).

    Parameters
    ----------
    latent_dim : int or ``"auto"``, optional
        Dimension of the latent space. If ``"auto"`` (default), will be set
        to ``input_dim // 2`` during build (with a minimum of 2).
    subnet : str, type, or keras.Layer, optional
        The subnet architecture. Can be ``"mlp"``, a class, or a Layer instance.
        Default is ``"mlp"``.
    subnet_kwargs : dict[str, any], optional
        Additional arguments for subnet construction. Default is None.
    **kwargs
        Additional arguments passed to the base Layer.
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
        latent_dim: int | str = "auto",
        subnet: str | type | keras.Layer = "mlp",
        subnet_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        self.latent_dim = latent_dim
        self.seed_generator = keras.random.SeedGenerator()

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "mlp":
            subnet_kwargs = Encoder.MLP_DEFAULT_CONFIG | subnet_kwargs

        self.subnet = find_network(subnet, **subnet_kwargs)
        self.mean_projector = None
        self.log_var_projector = None

    def build(self, input_shape: Shape) -> None:
        if self.built:
            return

        input_dim = input_shape[-1]

        # Auto-determine latent dimension if not specified
        if self.latent_dim == "auto":
            self.latent_dim = max(2, input_dim // 2)

        self.mean_projector = keras.layers.Dense(units=self.latent_dim, name="mean_projector")
        self.log_var_projector = keras.layers.Dense(units=self.latent_dim, name="log_var_projector")

        self.subnet.build(input_shape)
        subnet_output_shape = self.subnet.compute_output_shape(input_shape)

        self.mean_projector.build(subnet_output_shape)
        self.log_var_projector.build(subnet_output_shape)

    def call(self, x: Tensor, training: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """Encode input to latent distribution and sample.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., input_dim).
        training : bool, optional
            Whether in training mode. Default is False.

        Returns
        -------
        z : Tensor
            Sampled latent vector of shape (..., latent_dim).
        mean : Tensor
            Mean of the latent distribution.
        log_var : Tensor
            Log variance of the latent distribution.
        """
        h = self.subnet(x, training=training)
        mean = self.mean_projector(h, training=training)
        log_var = self.log_var_projector(h, training=training)

        # Reparameterization trick
        std = ops.exp(0.5 * log_var)
        epsilon = keras.random.normal(ops.shape(mean), seed=self.seed_generator)
        z = mean + std * epsilon

        return z, mean, log_var

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "latent_dim": self.latent_dim,
            "subnet": self.subnet,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
