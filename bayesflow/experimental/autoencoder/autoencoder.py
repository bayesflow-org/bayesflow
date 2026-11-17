import keras

from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs, layer_kwargs, weighted_mean
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.experimental")
class AutoEncoder(keras.Layer):
    """A standard (non-variational) autoencoder, which compresses data into a low-dimensional latent representation.
    This variant of autoencoder is trained only on a reconstruction loss. For a variational variant, see
    :class:`VariationalAutoEncoder`.
    """

    def __init__(
        self,
        latent_dim: int,
        encoder_network: keras.Layer,
        decoder_network: keras.Layer,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))
        self.latent_dim = latent_dim
        self.encoder_network = encoder_network
        self.encoder_projector = keras.layers.Dense(latent_dim, use_bias=False)
        self.decoder_network = decoder_network
        self.decoder_projector = None

    def build(self, input_shape):
        if self.built:
            return

        shape = input_shape
        self.encoder_network.build(shape)
        shape = self.encoder_network.compute_output_shape(shape)
        self.encoder_projector.build(shape)

        # ensure consistency in VAE
        shape = self.compute_output_shape(input_shape)

        self.decoder_network.build(shape)
        shape = self.decoder_network.compute_output_shape(shape)

        if self.decoder_projector is None:
            self.decoder_projector = keras.layers.Dense(units=input_shape[-1], use_bias=False)

        self.decoder_projector.build(shape)

    def compute_output_shape(self, input_shape):
        shape = input_shape
        shape = self.encoder_network.compute_output_shape(shape)
        shape = self.encoder_projector.compute_output_shape(shape)

        # ensure consistency in VAE
        shape = *shape[:-1], self.latent_dim

        return shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "latent_dim": self.latent_dim,
            "encoder_network": self.encoder_network,
            "decoder_network": self.decoder_network,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def call(self, xz: Tensor, training: bool = False, inverse: bool = False, **kwargs):
        if inverse:
            return self._inverse(xz, training=training, **kwargs)
        return self._forward(xz, training=training, **kwargs)

    def _forward(self, x: Tensor, training: bool = False, **kwargs):
        y = self.encoder_network(x, training=training, **filter_kwargs(kwargs, self.encoder_network.call))
        z = self.encoder_projector(y, training=training, **filter_kwargs(kwargs, self.encoder_projector.call))
        return z

    def _inverse(self, z: Tensor, training: bool = False, **kwargs):
        if self.decoder_projector is None:
            raise RuntimeError("Must call build before calling inverse.")

        y = self.decoder_network(z, training=training, **filter_kwargs(kwargs, self.decoder_network.call))
        x = self.decoder_projector(y, training=training, **filter_kwargs(kwargs, self.decoder_projector.call))
        return x

    def compute_metrics(
        self, x: Tensor, sample_weight: Tensor = None, stage: str = "training", **kwargs
    ) -> dict[str, Tensor]:
        training = stage == "training"
        z = self(x, training=training, inverse=False, **kwargs)
        reconstruction = self(z, training=training, inverse=True, **kwargs)
        loss = keras.ops.mean(keras.ops.square(x - reconstruction), axis=list(range(1, keras.ops.ndim(x))))
        loss = weighted_mean(loss, sample_weight)
        return {"loss": loss, "z": z}
