import keras

from bayesflow.types import Tensor
from bayesflow.metrics.functional import maximum_mean_discrepancy
from bayesflow.utils import resolve_seed, non_batch_axis, weighted_mean
from bayesflow.utils.serialization import serializable, serialize

from .autoencoder import AutoEncoder


@serializable("bayesflow.experimental")
class InfoVAE(AutoEncoder):
    """Information-Maximizing Variational Autoencoder.

    This implements the MMD version of the InfoVAE objective from Zhao et al.
    The loss is written as

        loss = reconstruction_loss
             + w_kl  * KL[q(z | x) || p(z)]
             + w_mmd * MMD[q(z), p(z)]

    Parameters
    ----------
    latent_dim
        Dimensionality of the latent variable.
    encoder_network
        Network mapping inputs to an encoder representation.
    decoder_network
        Network mapping latent samples to a decoder representation.
    alpha
        InfoVAE information parameter. Controls the weight of the conditional
        encoder KL through ``1 - alpha``.
    lambd
        InfoVAE aggregate-posterior matching parameter. Together with alpha,
        controls the MMD weight through ``alpha + lambda_ - 1``.
    mmd_kwargs
        Optional keyword arguments forwarded to ``maximum_mean_discrepancy``.
    """

    def __init__(
        self,
        latent_dim: int,
        encoder_network: keras.Layer,
        decoder_network: keras.Layer,
        alpha: float = 0.0,
        lambd: float = 1.0,
        mmd_kwargs: dict | None = None,
        **kwargs,
    ):
        super().__init__(
            latent_dim=latent_dim,
            encoder_network=encoder_network,
            decoder_network=decoder_network,
            **kwargs,
        )

        self.encoder_projector.units = 2 * latent_dim
        self.alpha = alpha
        self.lambd = lambd
        self.mmd_kwargs = mmd_kwargs or {}

        self.mmd_weight = self.alpha + self.lambd - 1.0
        self.kl_weight = 1.0 - self.alpha

        self.seed_generator = keras.random.SeedGenerator()

    def get_config(self):
        base_config = super().get_config()
        config = {
            "alpha": self.alpha,
            "lambd": self.lambd,
        }
        return base_config | serialize(config)

    def _encode(
        self,
        x: Tensor,
        training: bool = False,
        seed: int | keras.random.SeedGenerator | None = None,
        **kwargs,
    ):
        seed = resolve_seed(seed)

        z = super()._forward(x, training=training, **kwargs)
        mean, log_var = keras.ops.split(z, 2, axis=-1)

        epsilon = keras.random.normal(
            shape=keras.ops.shape(mean),
            seed=seed,
            dtype=mean.dtype,
        )

        sample = mean + keras.ops.exp(0.5 * log_var) * epsilon

        return z, mean, log_var, epsilon, sample

    def _forward(
        self,
        x: Tensor,
        training: bool = False,
        seed: int | keras.random.SeedGenerator | None = None,
        **kwargs,
    ):
        *_, sample = self._encode(
            x,
            training=training,
            seed=seed,
            **kwargs,
        )
        return sample

    def _conditional_kl(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """Per-example KL[q(z | x) || p(z)] for diagonal Gaussian q and standard normal p."""

        return 0.5 * keras.ops.sum(
            keras.ops.square(mean) + keras.ops.exp(log_var) - 1.0 - log_var,
            axis=non_batch_axis(mean),
        )

    def _reconstruction_loss(self, x: Tensor, reconstruction: Tensor) -> Tensor:
        """Per-example mean squared reconstruction error."""

        return keras.ops.mean(
            keras.ops.square(x - reconstruction),
            axis=non_batch_axis(x),
        )

    def compute_metrics(
        self,
        x: Tensor,
        sample_weight: Tensor = None,
        stage: str = "training",
        seed: int | keras.random.SeedGenerator | None = None,
        **kwargs,
    ) -> dict[str, Tensor]:
        training = stage == "training"
        seed = resolve_seed(seed)

        _, mean, log_var, _, sample = self._encode(
            x,
            training=training,
            seed=seed,
            **kwargs,
        )

        reconstruction = self(
            sample,
            training=training,
            inverse=True,
            **kwargs,
        )

        recon_loss = weighted_mean(self._reconstruction_loss(x, reconstruction), sample_weight)
        kl_loss = keras.ops.mean(self._conditional_kl(mean, log_var))

        loss = recon_loss + self.kl_weight * kl_loss

        if self.mmd_weight != 0.0:
            mmd_targets = keras.random.normal(shape=keras.ops.shape(sample), seed=seed, dtype=sample.dtype)

            marginal_mmd = maximum_mean_discrepancy(sample, mmd_targets, **self.mmd_kwargs)

            loss = loss + self.mmd_weight * marginal_mmd
        else:
            marginal_mmd = keras.ops.zeros((), dtype=sample.dtype)

        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss, "mmd_loss": marginal_mmd, "z": sample}
