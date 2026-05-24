import keras
from keras import ops

from bayesflow.types import Shape, Tensor
from bayesflow.utils import layer_kwargs, weighted_mean
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ..inference_network import InferenceNetwork
from .encoder import Encoder
from .decoder import Decoder


@serializable("bayesflow.networks")
class LatentInferenceNetwork(InferenceNetwork):
    """Latent-space inference network for amortized Bayesian inference.

    Parameters
    ----------
    inference_network : InferenceNetwork
        Pre-configured inference network operating in latent space.
    latent_shape : int, tuple, or ``"auto"``, optional
        Per-sample latent shape. Default ``"auto"`` (flat,
        ``max(2, prod(input_shape[1:]) // 2)``).
    latent_dim : int or ``"auto"``, optional
        Backward-compatible alias for a flat ``latent_shape``.
    encoder : str, type, or keras.Layer, optional
        Encoder subnet spec. Default ``"mlp"``.
    decoder : str, type, or keras.Layer, optional
        Decoder subnet spec. Default ``"mlp"``.
    encoder_kwargs, decoder_kwargs : dict, optional
        Extra arguments forwarded to the encoder/decoder subnets.
    kl_weight : float, optional
        Weight for KL loss. Default ``1e-3``.
    reconstruction_weight : float, optional
        Weight for reconstruction loss. Default ``1.0``.
    warmup_steps : int, optional
        Steps over which the inference-loss weight ramps from 0 to 1.
        Default ``1000``.
    **kwargs
        Forwarded to :py:class:`InferenceNetwork`.

    References
    ----------
    [1] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022).
    High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.

    Examples
    --------
    Flat latent:

    >>> from bayesflow.networks import DiffusionModel, LatentInferenceNetwork
    >>> lin = LatentInferenceNetwork(
    ...     inference_network=DiffusionModel(subnet_kwargs=dict(widths=(64, 64))),
    ...     latent_shape=8,
    ... )

    Spatial latent with UNet diffusion:

    >>> lin = LatentInferenceNetwork(
    ...     inference_network=DiffusionModel(subnet="unet"),
    ...     latent_shape=(7, 7, 4),
    ...     encoder=my_conv_downsampler,   # (B, 28, 28, 1) -> (B, 7, 7, *)
    ...     decoder=my_conv_upsampler,     # (B, 7, 7, 4)   -> (B, 28, 28, *)
    ... )
    """

    def __init__(
        self,
        *,
        inference_network: InferenceNetwork,
        latent_shape: int | tuple | str = None,
        latent_dim: int | str = "auto",
        encoder: str | type | keras.Layer = "mlp",
        decoder: str | type | keras.Layer = "mlp",
        encoder_kwargs: dict[str, any] = None,
        decoder_kwargs: dict[str, any] = None,
        kl_weight: float = 1e-3,
        reconstruction_weight: float = 1.0,
        warmup_steps: int = 1000,
        **kwargs,
    ):
        super().__init__(base_distribution="normal", **kwargs)

        if latent_shape is None:
            if encoder_kwargs is not None and "latent_shape" in encoder_kwargs:
                latent_shape = encoder_kwargs["latent_shape"]
            elif encoder_kwargs is not None and "latent_dim" in encoder_kwargs:
                latent_shape = encoder_kwargs["latent_dim"]
            else:
                latent_shape = latent_dim

        self._latent_shape_arg = latent_shape
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight
        self.warmup_steps = warmup_steps

        self._encoder_subnet = encoder
        self._encoder_kwargs = encoder_kwargs or {}
        self._decoder_subnet = decoder
        self._decoder_kwargs = decoder_kwargs or {}

        self.encoder = Encoder(
            latent_shape=latent_shape,
            subnet=encoder,
            subnet_kwargs=self._encoder_kwargs,
        )
        self.decoder = Decoder(
            subnet=decoder,
            subnet_kwargs=self._decoder_kwargs,
        )

        self.inference_network = inference_network

        self._training_steps = self.add_weight(
            name="training_steps",
            shape=(),
            initializer="zeros",
            trainable=False,
            dtype="int32",
        )

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        if self.built:
            return

        self.encoder.build(xz_shape)
        latent_per_sample = tuple(self.encoder.latent_shape)

        # Decoder reconstructs the full per-sample shape.
        self.decoder._output_shape_arg = tuple(xz_shape[1:])
        latent_full_shape = (xz_shape[0],) + latent_per_sample
        self.decoder.build(latent_full_shape)

        self.base_distribution.build(latent_full_shape)
        self.inference_network.build(latent_full_shape, conditions_shape)

    def encode(self, x: Tensor, training: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """Encode ``x`` to ``(z, mean, log_var)`` in latent space."""
        return self.encoder(x, training=training)

    def decode(self, z: Tensor, training: bool = False) -> Tensor:
        """Decode latent ``z`` back to input space."""
        return self.decoder(z, training=training)

    def _forward(
        self,
        x: Tensor,
        conditions: Tensor = None,
        density: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Encode ``x`` to latent space (no exact density)."""
        if density:
            raise NotImplementedError(
                "Exact density computation is not supported for LatentInferenceNetwork. "
                "Use sampling-based inference instead."
            )
        z, _, _ = self.encode(x, training=training)
        return z

    def _inverse(
        self,
        z: Tensor,
        conditions: Tensor = None,
        density: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Sample in latent space via the inner inference network, then decode."""
        if density:
            raise NotImplementedError(
                "Exact density computation is not supported for LatentInferenceNetwork. "
                "Use sampling-based inference instead."
            )
        z_denoised = self.inference_network._inverse(
            z, conditions=conditions, density=False, training=training, **kwargs
        )
        return self.decode(z_denoised, training=training)

    def compute_metrics(
        self,
        x: Tensor,
        conditions: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:
        """Combined loss: ``warmup * inference + recon_w * recon + kl_w * kl``."""
        training = stage == "training"

        if not self.built:
            xz_shape = ops.shape(x)
            conditions_shape = ops.shape(conditions) if conditions is not None else None
            self.build(xz_shape, conditions_shape)

        z, z_mean, z_log_var = self.encode(x, training=training)
        x_recon = self.decode(z, training=training)

        reconstruction_loss = self._compute_reconstruction_loss(x, x_recon)
        kl_loss = self._compute_kl_loss(z_mean, z_log_var)

        # Stop-gradient stabilises early training: inference loss does not
        # update the encoder until the latent geometry has settled.
        z_for_inference = ops.stop_gradient(z)
        inference_metrics = self.inference_network.compute_metrics(
            z_for_inference, conditions=conditions, sample_weight=sample_weight, stage=stage
        )
        inference_loss = inference_metrics["loss"]

        warmup_weight = self._compute_warmup_weight()

        loss = (
            warmup_weight * inference_loss
            + self.reconstruction_weight * weighted_mean(reconstruction_loss, sample_weight)
            + self.kl_weight * weighted_mean(kl_loss, sample_weight)
        )

        reconstruction_loss = weighted_mean(reconstruction_loss, sample_weight)
        kl_loss = weighted_mean(kl_loss, sample_weight)

        if training:
            self._training_steps.assign(self._training_steps + 1)

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "inference_loss": inference_loss,
            "warmup_weight": warmup_weight,
        }

    @staticmethod
    def _non_batch_axes(t: Tensor) -> list[int]:
        rank = len(ops.shape(t))
        return list(range(1, rank)) if rank > 1 else [-1]

    def _compute_reconstruction_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        return ops.mean(ops.square(x - x_recon), axis=self._non_batch_axes(x))

    def _compute_kl_loss(self, mean: Tensor, log_var: Tensor) -> Tensor:
        # KL(N(mean, var) || N(0, I)) summed over all latent dims, per-sample.
        per_elem = -0.5 * (1.0 + log_var - ops.square(mean) - ops.exp(log_var))
        return ops.sum(per_elem, axis=self._non_batch_axes(mean))

    def _compute_warmup_weight(self) -> Tensor:
        if self.warmup_steps <= 0:
            return ops.ones(())
        progress = ops.cast(self._training_steps, "float32") / ops.cast(self.warmup_steps, "float32")
        return ops.minimum(progress, 1.0)

    @property
    def latent_shape(self):
        return self.encoder.latent_shape if self.encoder.built else self._latent_shape_arg

    @property
    def latent_dim(self):
        if self.encoder.built:
            return self.encoder.latent_dim
        arg = self._latent_shape_arg
        return arg if isinstance(arg, int) else arg

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "latent_shape": self._latent_shape_arg,
            "encoder": self._encoder_subnet,
            "encoder_kwargs": self._encoder_kwargs,
            "decoder": self._decoder_subnet,
            "decoder_kwargs": self._decoder_kwargs,
            "inference_network": self.inference_network,
            "kl_weight": self.kl_weight,
            "reconstruction_weight": self.reconstruction_weight,
            "warmup_steps": self.warmup_steps,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
