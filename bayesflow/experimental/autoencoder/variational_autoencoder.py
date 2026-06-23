import keras

from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs, weighted_mean
from bayesflow.utils.serialization import serializable, serialize

from .autoencoder import AutoEncoder


@serializable("bayesflow.experimental")
class VariationalAutoEncoder(AutoEncoder):
    def __init__(
        self,
        latent_dim: int,
        encoder_network: keras.Layer,
        decoder_network: keras.Layer,
        kl_weight: float = 1e-6,
        recon_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(latent_dim, encoder_network, decoder_network, **kwargs)

        # adjust the number of output units to account for the split into mean and variance
        self.encoder_projector.units = 2 * latent_dim

        self.kl_weight = kl_weight
        self.recon_weight = recon_weight
        self.seed_generator = keras.random.SeedGenerator()

    def get_config(self):
        base_config = super().get_config()
        config = {
            "kl_weight": self.kl_weight,
            "recon_weight": self.recon_weight,
        }
        return base_config | serialize(config)

    def _encode(
        self, x: Tensor, training: bool = False, seed: int | keras.random.SeedGenerator | None = None, **kwargs
    ):
        if seed is None:
            seed = self.seed_generator

        y = self.encoder_network(x, training=training, **filter_kwargs(kwargs, self.encoder_network.call))
        z = self.encoder_projector(y, training=training, **filter_kwargs(kwargs, self.encoder_projector.call))
        mean, log_var = keras.ops.split(z, 2, axis=-1)
        epsilon = keras.random.normal(keras.ops.shape(mean), seed=seed, dtype=mean.dtype)
        sample = mean + keras.ops.exp(log_var / 2) * epsilon
        return z, mean, log_var, epsilon, sample

    def _forward(self, x: Tensor, training: bool = False, **kwargs):
        *_, sample = self._encode(x, training=training, **kwargs)
        return sample

    def compute_metrics(
        self, x: Tensor, sample_weight: Tensor = None, stage: str = "training", **kwargs
    ) -> dict[str, Tensor]:
        training = stage == "training"

        z, mean, log_var, epsilon, sample = self._encode(x, training=training, **kwargs)
        reconstruction = self(sample, training=training, inverse=True, **kwargs)

        non_batch_axes = list(range(1, keras.ops.ndim(mean)))
        recon_loss = self.recon_weight * keras.ops.mean(keras.ops.square(x - reconstruction), axis=non_batch_axes)
        kl_loss = self.kl_weight * keras.ops.sum(
            -0.5 * (1.0 + log_var - keras.ops.square(mean) - keras.ops.exp(log_var)), axis=non_batch_axes
        )
        loss = kl_loss + recon_loss
        loss = weighted_mean(loss, sample_weight)
        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss, "z": sample}
