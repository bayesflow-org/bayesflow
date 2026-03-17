import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import find_network, layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.networks")
class Decoder(keras.Layer):
    """Decoder network that maps latent vectors back to the original space.

    Can be used standalone or as part of a :py:class:`LatentDiffusionModel`.
    When used standalone, the decoder reconstructs data from a latent space
    produced by any inference network (e.g., ``FlowMatching``, ``DiffusionModel``).

    Parameters
    ----------
    output_dim : int or ``"auto"``, optional
        Dimension of the output space. If ``"auto"`` (default), must be set
        before building (e.g., by the parent ``LatentDiffusionModel``).
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
        output_dim: int | str = "auto",
        subnet: str | type | keras.Layer = "mlp",
        subnet_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))

        self.output_dim = output_dim

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "mlp":
            subnet_kwargs = Decoder.MLP_DEFAULT_CONFIG | subnet_kwargs

        self.subnet = find_network(subnet, **subnet_kwargs)
        self.output_projector = None

    def build(self, input_shape: Shape) -> None:
        if self.built:
            return

        if self.output_dim == "auto":
            raise ValueError("output_dim must be set before building the decoder.")

        self.output_projector = keras.layers.Dense(units=self.output_dim, name="output_projector")

        self.subnet.build(input_shape)
        subnet_output_shape = self.subnet.compute_output_shape(input_shape)
        self.output_projector.build(subnet_output_shape)

    def call(self, z: Tensor, training: bool = False) -> Tensor:
        """Decode latent vector to original space.

        Parameters
        ----------
        z : Tensor
            Latent tensor of shape (..., latent_dim).
        training : bool, optional
            Whether in training mode. Default is False.

        Returns
        -------
        Tensor
            Reconstructed output of shape (..., output_dim).
        """
        h = self.subnet(z, training=training)
        return self.output_projector(h, training=training)

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "output_dim": self.output_dim,
            "subnet": self.subnet,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
