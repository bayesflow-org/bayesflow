import math
from typing import Union, Literal

from keras import ops

from bayesflow.types import Tensor
from bayesflow.utils.serialization import deserialize, serializable

from .noise_schedule_base import NoiseSchedule


@serializable("bayesflow.experimental")
class CosineNoiseSchedule(NoiseSchedule):
    """Cosine noise schedule for diffusion models. This schedule is based on the cosine schedule from [1].

    [1] Diffusion Models Beat GANs on Image Synthesis: Dhariwal and Nichol (2022)
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
        self.log_snr_min = min_log_snr
        self.log_snr_max = max_log_snr

        self._t_min = self.get_t_from_log_snr(log_snr_t=self.log_snr_max, training=True)
        self._t_max = self.get_t_from_log_snr(log_snr_t=self.log_snr_min, training=True)

    def _truncated_t(self, t: Tensor) -> Tensor:
        return self._t_min + (self._t_max - self._t_min) * t

    def get_log_snr(self, t: Union[float, Tensor], training: bool) -> Tensor:
        """Get the log signal-to-noise ratio (lambda) for a given diffusion time."""
        t_trunc = self._truncated_t(t)
        return -2 * ops.log(ops.tan(math.pi * t_trunc * 0.5)) + 2 * self._shift

    def get_t_from_log_snr(self, log_snr_t: Union[Tensor, float], training: bool) -> Tensor:
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
        return dict(min_log_snr=self.log_snr_min, max_log_snr=self.log_snr_max, shift=self._shift)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class EDMNoiseSchedule(NoiseSchedule):
    """EDM noise schedule for diffusion models. This schedule is based on the EDM paper [1].
    This should be used with the F-prediction type in the diffusion model.

    [1] Elucidating the Design Space of Diffusion-Based Generative Models: Karras et al. (2022)
    """

    def __init__(self, sigma_data: float = 1.0, sigma_min: float = 1e-4, sigma_max: float = 80.0):
        """
        Initialize the EDM noise schedule.

        Parameters
        ----------
        sigma_data : float, optional
            The standard deviation of the output distribution. Input of the network is scaled by this factor and
            the weighting function is scaled by this factor as well.
        sigma_min : float, optional
            The minimum noise level. Only relevant for sampling. Default is 1e-4.
        sigma_max : float, optional
            The maximum noise level. Only relevant for sampling. Default is 80.0.
        """
        super().__init__(name="edm_noise_schedule", variance_type="preserving")
        self.sigma_data = sigma_data
        # training settings
        self.p_mean = -1.2
        self.p_std = 1.2
        # sampling settings
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.rho = 7

        # convert EDM parameters to signal-to-noise ratio formulation
        self.log_snr_min = -2 * ops.log(sigma_max)
        self.log_snr_max = -2 * ops.log(sigma_min)
        # t is not truncated for EDM by definition of the sampling schedule
        # training bounds should be set to avoid numerical issues
        self._log_snr_min_training = self.log_snr_min - 1  # one is never sampler during training
        self._log_snr_max_training = self.log_snr_max + 1  # 0 is almost surely never sampled during training

    def get_log_snr(self, t: Union[float, Tensor], training: bool) -> Tensor:
        """Get the log signal-to-noise ratio (lambda) for a given diffusion time."""
        if training:
            # SNR = -dist.icdf(t_trunc) # negative seems to be wrong in the paper in the Kingma paper
            loc = -2 * self.p_mean
            scale = 2 * self.p_std
            snr = loc + scale * ops.erfinv(2 * t - 1) * math.sqrt(2)
            snr = ops.clip(snr, x_min=self._log_snr_min_training, x_max=self._log_snr_max_training)
        else:  # sampling
            sigma_min_rho = self.sigma_min ** (1 / self.rho)
            sigma_max_rho = self.sigma_max ** (1 / self.rho)
            snr = -2 * self.rho * ops.log(sigma_max_rho + (1 - t) * (sigma_min_rho - sigma_max_rho))
        return snr

    def get_t_from_log_snr(self, log_snr_t: Union[float, Tensor], training: bool) -> Tensor:
        """Get the diffusion time (t) from the log signal-to-noise ratio (lambda)."""
        if training:
            # SNR = -dist.icdf(t_trunc) => t = dist.cdf(-snr)  # negative seems to be wrong in the Kingma paper
            loc = -2 * self.p_mean
            scale = 2 * self.p_std
            x = log_snr_t
            t = 0.5 * (1 + ops.erf((x - loc) / (scale * math.sqrt(2.0))))
        else:  # sampling
            # SNR = -2 * rho * log(sigma_max ** (1/rho) + (1 - t) * (sigma_min ** (1/rho) - sigma_max ** (1/rho)))
            # => t = 1 - ((exp(-snr/(2*rho)) - sigma_max ** (1/rho)) / (sigma_min ** (1/rho) - sigma_max ** (1/rho)))
            sigma_min_rho = self.sigma_min ** (1 / self.rho)
            sigma_max_rho = self.sigma_max ** (1 / self.rho)
            t = 1 - ((ops.exp(-log_snr_t / (2 * self.rho)) - sigma_max_rho) / (sigma_min_rho - sigma_max_rho))
        return t

    def derivative_log_snr(self, log_snr_t: Tensor, training: bool) -> Tensor:
        """Compute d/dt log(1 + e^(-snr(t))), which is used for the reverse SDE."""
        if training:
            raise NotImplementedError("Derivative of log SNR is not implemented for training mode.")
        # sampling mode
        t = self.get_t_from_log_snr(log_snr_t=log_snr_t, training=training)

        # SNR = -2*rho*log(s_max + (1 - x)*(s_min - s_max))
        s_max = self.sigma_max ** (1 / self.rho)
        s_min = self.sigma_min ** (1 / self.rho)
        u = s_max + (1 - t) * (s_min - s_max)
        # d/dx snr = 2*rho*(s_min - s_max) / u
        dsnr_dx = 2 * self.rho * (s_min - s_max) / u

        # Using the chain rule on f(t) = log(1 + e^(-snr(t))):
        # f'(t) = - (e^{-snr(t)} / (1 + e^{-snr(t)})) * dsnr_dt
        factor = ops.exp(-log_snr_t) / (1 + ops.exp(-log_snr_t))
        return -factor * dsnr_dx

    def get_weights_for_snr(self, log_snr_t: Tensor) -> Tensor:
        """Get weights for the signal-to-noise ratio (snr) for a given log signal-to-noise ratio (lambda)."""
        # for F-prediction: w = (ops.exp(-log_snr_t) + sigma_data^2) / (ops.exp(-log_snr_t)*sigma_data^2)
        return ops.exp(-log_snr_t) / ops.square(self.sigma_data) + 1

    def get_config(self):
        return dict(sigma_data=self.sigma_data, sigma_min=self.sigma_min, sigma_max=self.sigma_max)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
