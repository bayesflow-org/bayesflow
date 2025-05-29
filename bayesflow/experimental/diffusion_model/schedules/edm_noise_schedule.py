import math

from keras import ops

from bayesflow.types import Tensor
from bayesflow.utils.serialization import deserialize, serializable

from .noise_schedule import NoiseSchedule


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class EDMNoiseSchedule(NoiseSchedule):
    """EDM noise schedule for diffusion models. This schedule is based on the EDM paper [1].
    This should be used with the F-prediction type in the diffusion model.

    [1] Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). Elucidating the design space of diffusion-based
    generative models. Advances in Neural Information Processing Systems, 35, 26565-26577.
    """

    def __init__(self, sigma_data: float = 1.0, sigma_min: float = 1e-4, sigma_max: float = 80.0):
        """
        Initialize the EDM noise schedule.

        Parameters
        ----------
        sigma_data : float, optional
            The standard deviation of the output distribution. Input of the network is scaled by this factor and
            the weighting function is scaled by this factor as well. Default is 1.0.
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

    def get_log_snr(self, t: float | Tensor, training: bool) -> Tensor:
        """Get the log signal-to-noise ratio (lambda) for a given diffusion time."""
        if training:
            # SNR = -dist.icdf(t_trunc) # negative seems to be wrong in the Kingma paper
            loc = -2 * self.p_mean
            scale = 2 * self.p_std
            snr = loc + scale * ops.erfinv(2 * t - 1) * math.sqrt(2)
            snr = ops.clip(snr, x_min=self._log_snr_min_training, x_max=self._log_snr_max_training)
        else:
            sigma_min_rho = self.sigma_min ** (1 / self.rho)
            sigma_max_rho = self.sigma_max ** (1 / self.rho)
            snr = -2 * self.rho * ops.log(sigma_max_rho + (1 - t) * (sigma_min_rho - sigma_max_rho))
        return snr

    def get_t_from_log_snr(self, log_snr_t: float | Tensor, training: bool) -> Tensor:
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

    def derivative_log_snr(self, log_snr_t: Tensor, training: bool = False) -> Tensor:
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
        # for F-loss: w = (ops.exp(-log_snr_t) + sigma_data^2) / (ops.exp(-log_snr_t)*sigma_data^2)
        return 1 + ops.exp(-log_snr_t) / ops.square(self.sigma_data)

    def get_config(self):
        config = {"sigma_data": self.sigma_data, "sigma_min": self.sigma_min, "sigma_max": self.sigma_max}
        return config

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
