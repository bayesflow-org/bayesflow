import keras

from bayesflow.metrics.functional import maximum_mean_discrepancy
from bayesflow.types import Tensor
from bayesflow.utils import resolve_seed, non_batch_axis, weighted_mean
from bayesflow.utils.serialization import serializable, serialize
from .autoencoder import AutoEncoder


@serializable("bayesflow.experimental")
class VariationalAutoEncoder(AutoEncoder):
    """Information-Maximizing Variational Autoencoder according to [1].

    The loss is computed as

        loss = reconstruction_loss
             + w_kl  * KL[q(z | x) || p(z)]
             + w_mmd * MMD[q(z), p(z)]
    with
        w_kl  = 1 - alpha
        w_mmd = alpha + beta - 1

    Useful settings are:

        Vanilla VAE (default):  w_kl=1,    w_mmd=0  -> alpha=0,        beta=1
        beta-VAE:               w_kl=a,    w_mmd=0  -> alpha=1-a,      beta=a
        MMD/InfoVAE:            w_kl=0,    w_mmd=b  -> alpha=1,        beta=b
        Mixed objective:        w_kl=a,    w_mmd=b  -> alpha=1-a,      beta=a+b

    [1] Zhao, S., Song, J., & Ermon, S. (2019). InfoVAE: Balancing learning and
    inference in variational autoencoders. In Proceedings of the AAAI Conference on
    Artificial Intelligence (Vol. 33, No. 01, pp. 5885-5892).

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
    beta
        InfoVAE marginal distribution matching parameter. Together with alpha,
        controls the MMD weight through ``alpha + beta - 1``. In [1], this parameter
        is named lambda.
    mmd_kwargs
        Optional keyword arguments forwarded to ``maximum_mean_discrepancy``.
    """

    def __init__(
        self,
        latent_dim: int,
        encoder_network: keras.Layer,
        decoder_network: keras.Layer,
        alpha: float = 0.0,
        beta: float = 1.0,
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
        self.beta = beta
        self.mmd_kwargs = mmd_kwargs or {}

        self.seed_generator = keras.random.SeedGenerator()

    @property
    def kl_weight(self) -> float:
        return 1.0 - self.alpha

    @property
    def mmd_weight(self) -> float:
        return self.alpha + self.beta - 1.0

    def get_config(self):
        base_config = super().get_config()
        config = {"alpha": self.alpha, "lambd": self.beta, "mmd_kwargs": self.mmd_kwargs}
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

    def _marginal_mmd(self, sample: Tensor, seed: int | keras.random.SeedGenerator | None = None) -> Tensor:
        """MMD[q(z), p(z)] using samples from the aggregate posterior and prior."""

        targets = keras.random.normal(
            shape=keras.ops.shape(sample),
            seed=seed,
            dtype=sample.dtype,
        )

        return maximum_mean_discrepancy(sample, targets, **self.mmd_kwargs)

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
            mmd_loss = self._marginal_mmd(sample, seed=seed)
            loss = loss + self.mmd_weight * mmd_loss
        else:
            mmd_loss = keras.ops.zeros((), dtype=sample.dtype)

        return {"loss": loss, "recon_loss": recon_loss, "kl_loss": kl_loss, "mmd_loss": mmd_loss, "z": sample}
