import keras

from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize

from ..inference_network import InferenceNetwork
from ..diffusion import DiffusionModel
from ..latent import LatentInferenceNetwork


@serializable("bayesflow.networks")
class LatentDiffusionModel(LatentInferenceNetwork):
    """Latent Diffusion Model for amortized Bayesian inference.

    A thin wrapper around :py:class:`LatentInferenceNetwork` that
    defaults to a :py:class:`DiffusionModel` as the latent-space backbone.

    Accepts DiffusionModel-specific parameters (``diffusion_subnet``,
    ``noise_schedule``, etc.) and creates the DiffusionModel internally.
    Alternatively, pass any pre-configured :py:class:`InferenceNetwork` via
    ``inference_network`` to override the default.

    Parameters
    ----------
    latent_dim : int or ``"auto"``, optional
        Dimension of the latent space. Default is ``"auto"``.
    encoder : str, type, or keras.Layer, optional
        Encoder network specification. Default is ``"mlp"``.
    decoder : str, type, or keras.Layer, optional
        Decoder network specification. Default is ``"mlp"``.
    encoder_kwargs : dict[str, any], optional
        Additional arguments for encoder construction. Default is None.
    decoder_kwargs : dict[str, any], optional
        Additional arguments for decoder construction. Default is None.
    inference_network : InferenceNetwork or ``"auto"``, optional
        A pre-configured inference network. If ``"auto"`` (default), a
        ``DiffusionModel`` is created from the diffusion-specific parameters.
    diffusion_subnet : str, type, or keras.Layer, optional
        Network for diffusion noise prediction (only used when
        ``inference_network="auto"``). Default is ``"time_mlp"``.
    diffusion_subnet_kwargs : dict[str, any], optional
        Additional arguments for diffusion subnet. Default is None.
    noise_schedule : str, optional
        Noise schedule for diffusion. Default is ``"cosine"``.
    schedule_kwargs : dict[str, any], optional
        Additional arguments for noise schedule. Default is None.
    integrate_kwargs : dict[str, any], optional
        Configuration for ODE integration during sampling. Default is None.
    kl_weight : float, optional
        Weight for KL divergence loss. Default is 1e-3.
    reconstruction_weight : float, optional
        Weight for reconstruction loss. Default is 1.0.
    warmup_steps : int, optional
        Number of training steps to linearly increase inference loss weight
        from 0 to 1. Default is 1000.
    **kwargs
        Additional arguments passed to the base class.

    Examples
    --------
    Default usage with DiffusionModel:

    >>> ldm = LatentDiffusionModel(latent_dim=8)

    With FlowMatching as the latent-space inference network:

    >>> from bayesflow.networks import FlowMatching
    >>> ldm = LatentDiffusionModel(
    ...     latent_dim=8,
    ...     inference_network=FlowMatching(subnet_kwargs=dict(widths=(64, 64))),
    ... )
    """

    def __init__(
        self,
        *,
        latent_dim: int | str = "auto",
        encoder: str | type | keras.Layer = "mlp",
        decoder: str | type | keras.Layer = "mlp",
        encoder_kwargs: dict[str, any] = None,
        decoder_kwargs: dict[str, any] = None,
        inference_network: InferenceNetwork | str = "auto",
        diffusion_subnet: str | type | keras.Layer = "time_mlp",
        diffusion_subnet_kwargs: dict[str, any] = None,
        noise_schedule: str = "cosine",
        schedule_kwargs: dict[str, any] = None,
        integrate_kwargs: dict[str, any] = None,
        kl_weight: float = 1e-3,
        reconstruction_weight: float = 1.0,
        warmup_steps: int = 1000,
        **kwargs,
    ):
        _inference_network_config = inference_network

        # Create DiffusionModel when inference_network is "auto"
        if inference_network == "auto":
            inference_network = DiffusionModel(
                subnet=diffusion_subnet,
                subnet_kwargs=diffusion_subnet_kwargs,
                noise_schedule=noise_schedule,
                schedule_kwargs=schedule_kwargs,
                integrate_kwargs=integrate_kwargs,
            )
        elif isinstance(inference_network, str):
            raise ValueError(
                f"Unknown inference_network specification: {inference_network}. "
                f'Expected "auto" or an InferenceNetwork instance.'
            )

        super().__init__(
            inference_network=inference_network,
            latent_dim=latent_dim,
            encoder=encoder,
            decoder=decoder,
            encoder_kwargs=encoder_kwargs,
            decoder_kwargs=decoder_kwargs,
            kl_weight=kl_weight,
            reconstruction_weight=reconstruction_weight,
            warmup_steps=warmup_steps,
            **kwargs,
        )

        # Store diffusion-specific config for serialization
        self._diffusion_subnet = diffusion_subnet
        self._diffusion_subnet_kwargs = diffusion_subnet_kwargs
        self._noise_schedule = noise_schedule
        self._schedule_kwargs = schedule_kwargs
        self._integrate_kwargs = integrate_kwargs
        self._inference_network_config = _inference_network_config

    def get_config(self):
        base_config = super().get_config()

        config = {
            "inference_network": self._inference_network_config,
            "diffusion_subnet": self._diffusion_subnet,
            "diffusion_subnet_kwargs": self._diffusion_subnet_kwargs,
            "noise_schedule": self._noise_schedule,
            "schedule_kwargs": self._schedule_kwargs,
            "integrate_kwargs": self._integrate_kwargs,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
