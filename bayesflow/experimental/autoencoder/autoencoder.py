import keras

from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs, layer_kwargs, weighted_mean
from bayesflow.utils.serialization import deserialize, serializable, serialize


@serializable("bayesflow.experimental")
class AutoEncoder(keras.Layer):
    def __init__(
        self,
        latent_dim: int,
        encoder_network: keras.Layer,
        decoder_network: keras.Layer,
        projector_type: type = keras.layers.Dense,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))
        self.latent_dim = latent_dim
        self.encoder_network = encoder_network
        self.encoder_projector = keras.layers.Dense(latent_dim, use_bias=False, name="encoder_projector")
        self.decoder_network = decoder_network
        self.decoder_projector = None

    def build(self, input_shape):
        if self.built:
            return

        shape = input_shape
        self.encoder_network.build(shape)
        shape = self.encoder_network.compute_output_shape(shape)
        self.encoder_projector.build(shape)

        shape = (*input_shape[:-1], self.latent_dim)
        self.decoder_network.build(shape)
        shape = self.decoder_network.compute_output_shape(shape)

        if self.decoder_projector is None:
            self.decoder_projector = keras.layers.Dense(units=input_shape[-1], use_bias=False, name="decoder_projector")
        self.decoder_projector.build(shape)

        self.built = True
        for layer in self._flatten_layers():
            if not layer.built:
                print(layer.__class__.__name__, "not built")
            layer.built = True

    def compute_output_shape(self, input_shape):
        output_shape = *input_shape[:-1], self.latent_dim
        return output_shape

    def get_config(self):
        base_config = super().get_config()
        config = {
            "latent_dim": self.latent_dim,
            "encoder_network": self.encoder_network,
            "decoder_network": self.decoder_network,
            "decoder_projector": self.decoder_projector,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def call(self, xz, training=False, inverse: bool = False, **kwargs):
        if inverse:
            return self._inverse(xz, training=training, **kwargs)
        return self._forward(xz, training=training, **kwargs)

    def _forward(self, x, conditions=None, training=False, **kwargs):
        y = self.encoder_network(x, training=training, **filter_kwargs(kwargs, self.encoder_network.call))
        z = self.encoder_projector(y, training=training, **filter_kwargs(kwargs, self.encoder_projector.call))
        return z

    def _inverse(self, z, training=False, **kwargs):
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
