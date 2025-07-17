import math
from typing import Literal

from keras import ops

from bayesflow.types import Tensor
from bayesflow.utils.serialization import deserialize, serializable

from .noise_schedule import NoiseSchedule


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class CosineNoiseSchedule(NoiseSchedule):
    """Cosine noise schedule for diffusion models. This schedule is based on the cosine schedule from [1].

    A cosine schedule is a popular technique for controlling how the variance (noise level) or
    learning rate evolves during the training of diffusion models. It was proposed as an improvement
    over the original linear beta schedule in [2]

    [1] Dhariwal, P., & Nichol, A. (2021). Diffusion models beat gans on image synthesis.
    Advances in Neural Information Processing Systems, 34, 8780-8794.
    [2] Ho, J., Jain, A., & Abbeel, P. (2020). Denoising diffusion probabilistic models.
    Advances in Neural Information Processing Systems, 33, 6840-6851.
    """

    def __init__(
        self,
        min_log_snr: float = -15,
        max_log_snr: float = 15,
        shift: float = 0.0,
        weighting: Literal["sigmoid", "likelihood_weighting"] = "sigmoid",
    ):
        """
        Initialize the cosine noise schedule.

        Parameters
        ----------
        min_log_snr : float, optional
            The minimum log signal-to-noise ratio (lambda). Default is -15.
        max_log_snr : float, optional
            The maximum log signal-to-noise ratio (lambda). Default is 15.
        shift : float, optional
            Shift the log signal-to-noise ratio (lambda) by this amount. Default is 0.0.
            For images, use shift = log(base_resolution / d), where d is the used resolution of the image.
        weighting : Literal["sigmoid", "likelihood_weighting"], optional
            The type of weighting function to use for the noise schedule. Default is "sigmoid".
        """
        super().__init__(name="cosine_noise_schedule", variance_type="preserving", weighting=weighting)
        self._shift = shift
        self._weighting = weighting
        self.log_snr_min = min_log_snr
        self.log_snr_max = max_log_snr

        self._t_min = self.get_t_from_log_snr(log_snr_t=self.log_snr_max, training=True)
        self._t_max = self.get_t_from_log_snr(log_snr_t=self.log_snr_min, training=True)

    def _truncated_t(self, t: Tensor) -> Tensor:
        return self._t_min + (self._t_max - self._t_min) * t

    def get_log_snr(self, t: Tensor | float, training: bool) -> Tensor:
        """Get the log signal-to-noise ratio (lambda) for a given diffusion time."""
        t_trunc = self._truncated_t(t)
        return -2 * ops.log(ops.tan(math.pi * t_trunc * 0.5)) + 2 * self._shift

    def get_t_from_log_snr(self, log_snr_t: Tensor | float, training: bool) -> Tensor:
        """Get the diffusion time (t) from the log signal-to-noise ratio (lambda)."""
        # SNR = -2 * log(tan(pi*t/2)) => t = 2/pi * arctan(exp(-snr/2))
        return 2 / math.pi * ops.arctan(ops.exp((2 * self._shift - log_snr_t) * 0.5))

    def derivative_log_snr(self, log_snr_t: Tensor, training: bool) -> Tensor:
        """Compute d/dt log(1 + e^(-snr(t))), which is used for the reverse SDE."""
        t = self.get_t_from_log_snr(log_snr_t=log_snr_t, training=training)

        # Compute the truncated time t_trunc
        t_trunc = self._truncated_t(t)
        dsnr_dx = -(2 * math.pi) / ops.sin(math.pi * t_trunc)

        # Using the chain rule on f(t) = log(1 + e^(-snr(t))):
        # f'(t) = - (e^{-snr(t)} / (1 + e^{-snr(t)})) * dsnr_dt
        dsnr_dt = dsnr_dx * (self._t_max - self._t_min)
        factor = ops.exp(-log_snr_t) / (1 + ops.exp(-log_snr_t))
        return -factor * dsnr_dt

    def get_config(self):
        config = {
            "min_log_snr": self.log_snr_min,
            "max_log_snr": self.log_snr_max,
            "shift": self._shift,
            "weighting": self._weighting,
        }
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
