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
    """Latent-space inference network for Amortized Bayesian inference.

    Combines an autoencoder (encoder/decoder) with any inference network
    operating in the learned latent space. The model approximates posterior
    distributions by performing inference in a compressed latent representation.

    Any :py:class:`InferenceNetwork` subclass can be used as the latent-space
    backbone — e.g., :py:class:`DiffusionModel`, :py:class:`FlowMatching`,
    :py:class:`CouplingFlow`, :py:class:`ConsistencyModel`.

    The training objective combines:

    - **Inference loss**: the loss from the latent-space inference network
    - **Reconstruction loss**: MSE between input and reconstruction
    - **KL loss**: regularization of latent space toward N(0, I)

    A warmup schedule gradually increases the inference loss weight to
    ensure stable training.

    Parameters
    ----------
    inference_network : InferenceNetwork
        A pre-configured inference network to operate in latent space.
    latent_dim : int or ``"auto"``, optional
        Dimension of the latent space. If ``"auto"`` (default), the encoder
        will auto-determine based on input dimensions.
    encoder : str, type, or keras.Layer, optional
        Encoder network specification. Default is ``"mlp"``.
    decoder : str, type, or keras.Layer, optional
        Decoder network specification. Default is ``"mlp"``.
    encoder_kwargs : dict[str, any], optional
        Additional arguments for encoder construction. Default is None.
    decoder_kwargs : dict[str, any], optional
        Additional arguments for decoder construction. Default is None.
    kl_weight : float, optional
        Weight for KL divergence loss. Default is 1e-3.
    reconstruction_weight : float, optional
        Weight for reconstruction loss. Default is 1.0.
    warmup_steps : int, optional
        Number of training steps to linearly increase inference loss weight
        from 0 to 1. Default is 1000.
    **kwargs
        Additional arguments passed to the base InferenceNetwork.

    References
    ----------
    [1] Rombach, R., Blattmann, A., Lorenz, D., Esser, P., & Ommer, B. (2022).
    High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022.

    Examples
    --------
    With DiffusionModel:

    >>> from bayesflow.networks import DiffusionModel
    >>> lin = LatentInferenceNetwork(
    ...     inference_network=DiffusionModel(subnet_kwargs=dict(widths=(64, 64))),
    ...     latent_dim=8,
    ... )

    With FlowMatching:

    >>> from bayesflow.networks import FlowMatching
    >>> lin = LatentInferenceNetwork(
    ...     inference_network=FlowMatching(subnet_kwargs=dict(widths=(64, 64))),
    ...     latent_dim=8,
    ... )

    Standalone encoder/decoder usage:

    >>> from bayesflow.networks.inference.latent import Encoder, Decoder
    >>> encoder = Encoder(latent_dim=8)
    >>> decoder = Decoder(output_dim=16)
    """

    def __init__(
        self,
        *,
        inference_network: InferenceNetwork,
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

        if latent_dim == "auto":
            if encoder_kwargs is not None and "latent_dim" in encoder_kwargs:
                latent_dim = encoder_kwargs["latent_dim"]

        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.reconstruction_weight = reconstruction_weight
        self.warmup_steps = warmup_steps

        # Store configs for serialization
        self._encoder_subnet = encoder
        self._encoder_kwargs = encoder_kwargs or {}
        self._decoder_subnet = decoder
        self._decoder_kwargs = decoder_kwargs or {}

        # Autoencoder components
        self.encoder = Encoder(
            latent_dim=latent_dim,
            subnet=encoder,
            subnet_kwargs=self._encoder_kwargs,
        )
        self.decoder = Decoder(
            subnet=decoder,
            subnet_kwargs=self._decoder_kwargs,
        )

        # Latent-space inference network
        self.inference_network = inference_network

        # Training step counter for warmup
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

        input_dim = xz_shape[-1]

        # Build encoder
        self.encoder.build(xz_shape)
        actual_latent_dim = self.encoder.latent_dim

        # Build decoder with correct output dimension
        self.decoder.output_dim = input_dim
        latent_shape = tuple(xz_shape[:-1]) + (actual_latent_dim,)
        self.decoder.build(latent_shape)

        # Build base distribution for latent space
        self.base_distribution.build(latent_shape)

        # Build the internal inference network for latent space
        self.inference_network.build(latent_shape, conditions_shape)

    def encode(self, x: Tensor, training: bool = False) -> tuple[Tensor, Tensor, Tensor]:
        """Encode input to latent space.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., input_dim).
        training : bool, optional
            Whether in training mode. Default is False.

        Returns
        -------
        z : Tensor
            Sampled latent vector.
        mean : Tensor
            Mean of latent distribution.
        log_var : Tensor
            Log variance of latent distribution.
        """
        return self.encoder(x, training=training)

    def decode(self, z: Tensor, training: bool = False) -> Tensor:
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
            Reconstructed output.
        """
        return self.decoder(z, training=training)

    def _forward(
        self,
        x: Tensor,
        conditions: Tensor = None,
        density: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Forward pass: encode to latent space.

        Note: Exact density computation is not supported for latent inference networks.
        """
        z, _, _ = self.encode(x, training=training)

        if density:
            raise NotImplementedError(
                "Exact density computation is not supported for LatentInferenceNetwork. "
                "Use sampling-based inference instead."
            )

        return z

    def _inverse(
        self,
        z: Tensor,
        conditions: Tensor = None,
        density: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Inverse pass: denoise latent via inference network and decode to original space."""
        if density:
            raise NotImplementedError(
                "Exact density computation is not supported for LatentInferenceNetwork. "
                "Use sampling-based inference instead."
            )

        # Use the internal inference network to denoise z
        z_denoised = self.inference_network._inverse(
            z, conditions=conditions, density=False, training=training, **kwargs
        )

        # Decode to original space
        x = self.decode(z_denoised, training=training)

        return x

    def compute_metrics(
        self,
        x: Tensor,
        conditions: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:
        """Compute training metrics and loss.

        The total loss combines:
        - Inference loss (from the latent-space inference network)
        - Reconstruction loss (MSE between input and reconstruction)
        - KL loss (regularization toward N(0, I))

        A warmup schedule gradually increases the inference loss weight.

        Parameters
        ----------
        x : Tensor
            Input tensor (parameters to infer).
        conditions : Tensor, optional
            Conditioning information from summary network. Default is None.
        sample_weight : Tensor, optional
            Per-sample weights. Default is None.
        stage : str, optional
            Training stage (``"training"``, ``"validation"``). Default is ``"training"``.

        Returns
        -------
        dict[str, Tensor]
            Dictionary containing ``"loss"`` and component losses.
        """
        training = stage == "training"

        if not self.built:
            xz_shape = ops.shape(x)
            conditions_shape = ops.shape(conditions) if conditions is not None else None
            self.build(xz_shape, conditions_shape)

        # Encode
        z, z_mean, z_log_var = self.encode(x, training=training)

        # Reconstruct
        x_recon = self.decode(z, training=training)

        # Compute individual losses
        reconstruction_loss = self._compute_reconstruction_loss(x, x_recon)
        kl_loss = self._compute_kl_loss(z_mean, z_log_var)

        # Detach encoder from inference loss to stabilize training
        z_for_inference = ops.stop_gradient(z)

        # Use the inference network's compute_metrics for the latent loss
        inference_metrics = self.inference_network.compute_metrics(
            z_for_inference, conditions=conditions, sample_weight=sample_weight, stage=stage
        )
        inference_loss = inference_metrics["loss"]

        # Warmup weight for inference loss
        warmup_weight = self._compute_warmup_weight()

        # Combined loss
        loss = (
            warmup_weight * inference_loss
            + self.reconstruction_weight * weighted_mean(reconstruction_loss, sample_weight)
            + self.kl_weight * weighted_mean(kl_loss, sample_weight)
        )

        reconstruction_loss = weighted_mean(reconstruction_loss, sample_weight)
        kl_loss = weighted_mean(kl_loss, sample_weight)

        # Update training step counter
        if training:
            self._training_steps.assign(self._training_steps + 1)

        return {
            "loss": loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss,
            "inference_loss": inference_loss,
            "warmup_weight": warmup_weight,
        }

    def _compute_reconstruction_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Compute mean squared error reconstruction loss."""
        return ops.mean(ops.square(x - x_recon), axis=-1)

    def _compute_kl_loss(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """Compute KL divergence from N(mean, var) to N(0, I)."""
        # KL = -0.5 * sum(1 + log_var - mean^2 - var)
        kl = -0.5 * ops.sum(1.0 + log_var - ops.square(mean) - ops.exp(log_var), axis=-1)
        return kl

    def _compute_warmup_weight(self) -> Tensor:
        """Compute warmup weight for inference loss (linear schedule)."""
        if self.warmup_steps <= 0:
            return ops.ones(())

        progress = ops.cast(self._training_steps, "float32") / ops.cast(self.warmup_steps, "float32")
        return ops.minimum(progress, 1.0)

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "latent_dim": self.latent_dim,
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
