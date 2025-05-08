import math
from abc import ABC, abstractmethod
from typing import Union, Literal

from keras import ops

from bayesflow.types import Tensor
from bayesflow.utils.serialization import deserialize, serializable


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class NoiseSchedule(ABC):
    r"""Noise schedule for diffusion models. We follow the notation from [1].

    The diffusion process is defined by a noise schedule, which determines how the noise level changes over time.
    We define the noise schedule as a function of the log signal-to-noise ratio (lambda), which can be
    interchangeably used with the diffusion time (t).

    The noise process is defined as: z = alpha(t) * x + sigma(t) * e, where e ~ N(0, I).
    The schedule is defined as: \lambda(t) = \log \sigma^2(t) - \log \alpha^2(t).

    We can also define a weighting function for each noise level for the loss function. Often the noise schedule is
    the same for the forward and reverse process, but this is not necessary and can be changed via the training flag.

    [1] Variational Diffusion Models 2.0: Understanding Diffusion Model Objectives as the ELBO with Simple Data
    Augmentation: Kingma et al. (2023)
    """

    def __init__(
        self,
        name: str,
        variance_type: Literal["preserving", "exploding"],
        weighting: Literal["sigmoid", "likelihood_weighting"] = None,
    ):
        """
        Initialize the noise schedule.

        Parameters
        ----------
        name : str
            The name of the noise schedule.
        variance_type : Literal["preserving", "exploding"]
            If the variance of noise added to the data should be preserved over time, use "preserving".
            If the variance of noise added to the data should increase over time, use "exploding".
            Default is "preserving".
        weighting : Literal["sigmoid", "likelihood_weighting"], optional
            The type of weighting function to use for the noise schedule.
            Default is None, which means no weighting is applied.
        """
        self.name = name
        self._variance_type = variance_type
        self.log_snr_min = None  # should be set in the subclasses
        self.log_snr_max = None  # should be set in the subclasses
        self._weighting = weighting

    @abstractmethod
    def get_log_snr(self, t: Union[float, Tensor], training: bool) -> Tensor:
        """Get the log signal-to-noise ratio (lambda) for a given diffusion time."""
        pass

    @abstractmethod
    def get_t_from_log_snr(self, log_snr_t: Union[float, Tensor], training: bool) -> Tensor:
        """Get the diffusion time (t) from the log signal-to-noise ratio (lambda)."""
        pass

    @abstractmethod
    def derivative_log_snr(self, log_snr_t: Union[float, Tensor], training: bool) -> Tensor:
        r"""Compute \beta(t) = d/dt log(1 + e^(-snr(t))). This is usually used for the reverse SDE."""
        pass

    def get_drift_diffusion(self, log_snr_t: Tensor, x: Tensor = None, training: bool = False) -> tuple[Tensor, Tensor]:
        r"""Compute the drift and optionally the squared diffusion term for the reverse SDE.
        It can be derived from the derivative of the schedule:

        .. math::
            \beta(t) = d/dt \log(1 + e^{-snr(t)})

            f(z, t) = -0.5 * \beta(t) * z

            g(t)^2 = \beta(t)

        The corresponding differential equations are::

            SDE: d(z) = [ f(z, t) - g(t)^2 * score(z, lambda) ] dt + g(t) dW
            ODE: dz = [ f(z, t) - 0.5 * g(t)^2 * score(z, lambda) ] dt

        For a variance exploding schedule, one should set f(z, t) = 0.
        """
        beta = self.derivative_log_snr(log_snr_t=log_snr_t, training=training)
        if x is None:  # return g^2 only
            return beta
        if self._variance_type == "preserving":
            f = -0.5 * beta * x
        elif self._variance_type == "exploding":
            f = ops.zeros_like(beta)
        else:
            raise ValueError(f"Unknown variance type: {self._variance_type}")
        return f, beta

    def get_alpha_sigma(self, log_snr_t: Tensor, training: bool) -> tuple[Tensor, Tensor]:
        """Get alpha and sigma for a given log signal-to-noise ratio (lambda).

        Default is a variance preserving schedule::

            alpha(t) = sqrt(sigmoid(log_snr_t))
            sigma(t) = sqrt(sigmoid(-log_snr_t))

        For a variance exploding schedule, one should set alpha^2 = 1 and sigma^2 = exp(-lambda)
        """
        if self._variance_type == "preserving":
            # variance preserving schedule
            alpha_t = ops.sqrt(ops.sigmoid(log_snr_t))
            sigma_t = ops.sqrt(ops.sigmoid(-log_snr_t))
        elif self._variance_type == "exploding":
            # variance exploding schedule
            alpha_t = ops.ones_like(log_snr_t)
            sigma_t = ops.sqrt(ops.exp(-log_snr_t))
        else:
            raise TypeError(f"Unknown variance type: {self._variance_type}")
        return alpha_t, sigma_t

    def get_weights_for_snr(self, log_snr_t: Tensor) -> Tensor:
        """Get weights for the signal-to-noise ratio (snr) for a given log signal-to-noise ratio (lambda).
        Default weighting is None, which means only ones are returned.
        Generally, weighting functions should be defined for a noise prediction loss.
        """
        if self._weighting is None:
            return ops.ones_like(log_snr_t)
        elif self._weighting == "sigmoid":
            # sigmoid weighting based on Kingma et al. (2023)
            return ops.sigmoid(-log_snr_t + 2)
        elif self._weighting == "likelihood_weighting":
            # likelihood weighting based on Song et al. (2021)
            g_squared = self.get_drift_diffusion(log_snr_t=log_snr_t)
            sigma_t = self.get_alpha_sigma(log_snr_t=log_snr_t, training=True)[1]
            return g_squared / ops.square(sigma_t)
        else:
            raise TypeError(f"Unknown weighting type: {self._weighting}")

    def get_config(self):
        return dict(name=self.name, variance_type=self._variance_type)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def validate(self):
        """Validate the noise schedule."""
        if self.log_snr_min >= self.log_snr_max:
            raise ValueError("min_log_snr must be less than max_log_snr.")
        for training in [True, False]:
            if not ops.isfinite(self.get_log_snr(0.0, training=training)):
                raise ValueError(f"log_snr(0) must be finite with training={training}.")
            if not ops.isfinite(self.get_log_snr(1.0, training=training)):
                raise ValueError(f"log_snr(1) must be finite with training={training}.")
            if not ops.isfinite(self.get_t_from_log_snr(self.log_snr_max, training=training)):
                raise ValueError(f"t(0) must be finite with training={training}.")
            if not ops.isfinite(self.get_t_from_log_snr(self.log_snr_min, training=training)):
                raise ValueError(f"t(1) must be finite with training={training}.")
        if not ops.isfinite(self.derivative_log_snr(self.log_snr_max, training=False)):
            raise ValueError("dt/t log_snr(0) must be finite.")
        if not ops.isfinite(self.derivative_log_snr(self.log_snr_min, training=False)):
            raise ValueError("dt/t log_snr(1) must be finite.")


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
