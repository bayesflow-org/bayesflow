from collections.abc import Sequence
from abc import ABC, abstractmethod
import keras
from keras import ops

from bayesflow.utils.serialization import serialize, deserialize, serializable
from bayesflow.types import Tensor, Shape
import bayesflow as bf
from bayesflow.networks import InferenceNetwork
import math

from bayesflow.utils import (
    expand_right_as,
    find_network,
    jacobian_trace,
    layer_kwargs,
    weighted_mean,
    integrate,
)


@serializable
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

    def __init__(self, name: str, variance_type: str):
        self.name = name
        self.variance_type = variance_type  # 'exploding' or 'preserving'
        self._log_snr_min = -15  # should be set in the subclasses
        self._log_snr_max = 15  # should be set in the subclasses

    @property
    def scale_base_distribution(self):
        """Get the scale of the base distribution."""
        if self.variance_type == "preserving":
            return 1.0
        elif self.variance_type == "exploding":
            # e.g., EDM is a variance exploding schedule
            return ops.exp(-self._log_snr_min)
        else:
            raise ValueError(f"Unknown variance type: {self.variance_type}")

    @abstractmethod
    def get_log_snr(self, t: Tensor, training: bool) -> Tensor:
        """Get the log signal-to-noise ratio (lambda) for a given diffusion time."""
        pass

    @abstractmethod
    def get_t_from_log_snr(self, log_snr_t: Tensor, training: bool) -> Tensor:
        """Get the diffusion time (t) from the log signal-to-noise ratio (lambda)."""
        pass

    @abstractmethod
    def derivative_log_snr(self, log_snr_t: Tensor, training: bool) -> Tensor:
        r"""Compute \beta(t) = d/dt log(1 + e^(-snr(t))). This is usually used for the reverse SDE."""
        pass

    def get_drift_diffusion(self, log_snr_t: Tensor, x: Tensor = None, training: bool = True) -> tuple[Tensor, Tensor]:
        r"""Compute the drift and optionally the diffusion term for the reverse SDE.
        Usually it can be derived from the derivative of the schedule:
            \beta(t) = d/dt log(1 + e^(-snr(t)))
            f(z, t) = -0.5 * \beta(t) * z
            g(t)^2 = \beta(t)

            SDE: d(z) = [ f(z, t) - g(t)^2 * score(z, lambda) ] dt + g(t) dW
            ODE: dz = [ f(z, t) - 0.5 * g(t)^2 * score(z, lambda) ] dt

        For a variance exploding schedule, one should set f(z, t) = 0.
        """
        # Default implementation is to return the diffusion term only
        beta = self.derivative_log_snr(log_snr_t=log_snr_t, training=training)
        if x is None:  # return g only
            return ops.sqrt(beta)
        if self.variance_type == "preserving":
            f = -0.5 * beta * x
        elif self.variance_type == "exploding":
            f = ops.zeros_like(beta)
        else:
            raise ValueError(f"Unknown variance type: {self.variance_type}")
        return f, ops.sqrt(beta)

    def get_alpha_sigma(self, log_snr_t: Tensor, training: bool) -> tuple[Tensor, Tensor]:
        """Get alpha and sigma for a given log signal-to-noise ratio (lambda).

        Default is a variance preserving schedule:
            alpha(t) = sqrt(sigmoid(log_snr_t))
            sigma(t) = sqrt(sigmoid(-log_snr_t))
        For a variance exploding schedule, one should set alpha^2 = 1 and sigma^2 = exp(-lambda)
        """
        if self.variance_type == "preserving":
            # variance preserving schedule
            alpha_t = keras.ops.sqrt(keras.ops.sigmoid(log_snr_t))
            sigma_t = keras.ops.sqrt(keras.ops.sigmoid(-log_snr_t))
        elif self.variance_type == "exploding":
            # variance exploding schedule
            alpha_t = ops.ones_like(log_snr_t)
            sigma_t = ops.sqrt(ops.exp(-log_snr_t))
        else:
            raise ValueError(f"Unknown variance type: {self.variance_type}")
        return alpha_t, sigma_t

    def get_weights_for_snr(self, log_snr_t: Tensor) -> Tensor:
        """Get weights for the signal-to-noise ratio (snr) for a given log signal-to-noise ratio (lambda). Default is 1.
        Generally, weighting functions should be defined for a noise prediction loss.
        """
        # sigmoid: ops.sigmoid(-log_snr_t / 2), based on Kingma et al. (2023)
        # min-snr with gamma = 5, based on Hang et al. (2023)
        # 1 / ops.cosh(log_snr_t / 2) * ops.minimum(ops.ones_like(log_snr_t), gamma * ops.exp(-log_snr_t))
        return ops.ones_like(log_snr_t)

    def get_config(self):
        return dict(name=self.name, variance_type=self.variance_type)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


@serializable
class LinearNoiseSchedule(NoiseSchedule):
    """Linear noise schedule for diffusion models.

    The linear noise schedule with likelihood weighting is based on [1].

    [1] Maximum Likelihood Training of Score-Based Diffusion Models: Song et al. (2021)
    """

    def __init__(self, min_log_snr: float = -15, max_log_snr: float = 15):
        super().__init__(name="linear_noise_schedule")
        self._log_snr_min = min_log_snr
        self._log_snr_max = max_log_snr

        self._t_min = self.get_t_from_log_snr(log_snr_t=self._log_snr_max, training=True)
        self._t_max = self.get_t_from_log_snr(log_snr_t=self._log_snr_max, training=True)

    def get_log_snr(self, t: Tensor, training: bool) -> Tensor:
        """Get the log signal-to-noise ratio (lambda) for a given diffusion time."""
        t_trunc = self._t_min + (self._t_max - self._t_min) * t
        # SNR = -log(exp(t^2) - 1)
        return -ops.log(ops.exp(ops.square(t_trunc)) - 1)

    def get_t_from_log_snr(self, log_snr_t: Tensor, training: bool) -> Tensor:
        """Get the diffusion time (t) from the log signal-to-noise ratio (lambda)."""
        # SNR = -log(exp(t^2) - 1) => t = sqrt(log(1 + exp(-snr)))
        return ops.sqrt(ops.log(1 + ops.exp(-log_snr_t)))

    def derivative_log_snr(self, log_snr_t: Tensor, training: bool) -> Tensor:
        """Compute d/dt log(1 + e^(-snr(t))), which is used for the reverse SDE."""
        t = self.get_t_from_log_snr(log_snr_t=log_snr_t, training=training)

        # Compute the truncated time t_trunc
        t_trunc = self._t_min + (self._t_max - self._t_min) * t
        dsnr_dx = -(2 * t_trunc * ops.exp(t_trunc**2)) / (ops.exp(t_trunc**2) - 1)

        # Using the chain rule on f(t) = log(1 + e^(-snr(t))):
        # f'(t) = - (e^{-snr(t)} / (1 + e^{-snr(t)})) * dsnr_dt
        dsnr_dt = dsnr_dx * (self._t_max - self._t_min)
        factor = ops.exp(-log_snr_t) / (1 + ops.exp(-log_snr_t))
        return -factor * dsnr_dt

    def get_weights_for_snr(self, log_snr_t: Tensor) -> Tensor:
        """Get weights for the signal-to-noise ratio (snr) for a given log signal-to-noise ratio (lambda).
        Default is the likelihood weighting based on Song et al. (2021).
        """
        g = self.get_drift_diffusion(log_snr_t=log_snr_t)
        sigma_t = self.get_alpha_sigma(log_snr_t=log_snr_t, training=True)[1]
        return ops.square(g / sigma_t)

    def get_config(self):
        return dict(min_log_snr=self._log_snr_min, max_log_snr=self._log_snr_max)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


@serializable
class CosineNoiseSchedule(NoiseSchedule):
    """Cosine noise schedule for diffusion models. This schedule is based on the cosine schedule from [1].
    For images, use s_shift_cosine = log(base_resolution / d), where d is the used resolution of the image.

    [1] Diffusion models beat gans on image synthesis: Dhariwal and Nichol (2022)
    """

    def __init__(self, min_log_snr: float = -15, max_log_snr: float = 15, s_shift_cosine: float = 0.0):
        super().__init__(name="cosine_noise_schedule", variance_type="preserving")
        self._s_shift_cosine = s_shift_cosine
        self._log_snr_min = min_log_snr
        self._log_snr_max = max_log_snr
        self._s_shift_cosine = s_shift_cosine

        self._t_min = self.get_t_from_log_snr(log_snr_t=self._log_snr_max, training=True)
        self._t_max = self.get_t_from_log_snr(log_snr_t=self._log_snr_max, training=True)

    def get_log_snr(self, t: Tensor, training: bool) -> Tensor:
        """Get the log signal-to-noise ratio (lambda) for a given diffusion time."""
        t_trunc = self._t_min + (self._t_max - self._t_min) * t
        # SNR = -2 * log(tan(pi*t/2))
        return -2 * ops.log(ops.tan(math.pi * t_trunc / 2)) + 2 * self._s_shift_cosine

    def get_t_from_log_snr(self, log_snr_t: Tensor, training: bool) -> Tensor:
        """Get the diffusion time (t) from the log signal-to-noise ratio (lambda)."""
        # SNR = -2 * log(tan(pi*t/2)) => t = 2/pi * arctan(exp(-snr/2))
        return 2 / math.pi * ops.arctan(ops.exp((2 * self._s_shift_cosine - log_snr_t) / 2))

    def derivative_log_snr(self, log_snr_t: Tensor, training: bool) -> Tensor:
        """Compute d/dt log(1 + e^(-snr(t))), which is used for the reverse SDE."""
        t = self.get_t_from_log_snr(log_snr_t=log_snr_t, training=training)

        # Compute the truncated time t_trunc
        t_trunc = self._t_min + (self._t_max - self._t_min) * t
        dsnr_dx = -(2 * math.pi) / ops.sin(math.pi * t_trunc)

        # Using the chain rule on f(t) = log(1 + e^(-snr(t))):
        # f'(t) = - (e^{-snr(t)} / (1 + e^{-snr(t)})) * dsnr_dt
        dsnr_dt = dsnr_dx * (self._t_max - self._t_min)
        factor = ops.exp(-log_snr_t) / (1 + ops.exp(-log_snr_t))
        return -factor * dsnr_dt

    def get_weights_for_snr(self, log_snr_t: Tensor) -> Tensor:
        """Get weights for the signal-to-noise ratio (snr) for a given log signal-to-noise ratio (lambda).
        Default is the sigmoid weighting based on Kingma et al. (2023).
        """
        return ops.sigmoid(-log_snr_t / 2)

    def get_config(self):
        return dict(min_log_snr=self._log_snr_min, max_log_snr=self._log_snr_max, s_shift_cosine=self._s_shift_cosine)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))


@serializable
class EDMNoiseSchedule(NoiseSchedule):
    """EDM noise schedule for diffusion models. This schedule is based on the EDM paper [1].

    [1] Elucidating the Design Space of Diffusion-Based Generative Models: Karras et al. (2022)
    """

    def __init__(self, sigma_data: float = 0.5, sigma_min: float = 0.002, sigma_max: float = 80):
        super().__init__(name="edm_noise_schedule", variance_type="exploding")
        super().__init__(name="edm_noise_schedule")
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.p_mean = -1.2
        self.p_std = 1.2
        self.rho = 7

        # convert EDM parameters to signal-to-noise ratio formulation
        self._log_snr_min = -2 * ops.log(sigma_max)
        self._log_snr_max = -2 * ops.log(sigma_min)
        self._t_min = self.get_t_from_log_snr(log_snr_t=self._log_snr_max, training=True)
        self._t_max = self.get_t_from_log_snr(log_snr_t=self._log_snr_max, training=True)

    def get_log_snr(self, t: Tensor, training: bool) -> Tensor:
        """Get the log signal-to-noise ratio (lambda) for a given diffusion time."""
        t_trunc = self._t_min + (self._t_max - self._t_min) * t
        if training:
            # SNR = -dist.icdf(t_trunc)
            loc = -2 * self.p_mean
            scale = 2 * self.p_std
            x = t_trunc
            snr = -(loc + scale * ops.erfinv(2 * x - 1) * math.sqrt(2))
            snr = keras.ops.clip(snr, x_min=self._log_snr_min, x_max=self._log_snr_max)
        else:  # sampling
            snr = (
                -2
                * self.rho
                * ops.log(
                    self.sigma_max ** (1 / self.rho)
                    + (1 - t_trunc) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
                )
            )
        return snr

    def get_t_from_log_snr(self, log_snr_t: Tensor, training: bool) -> Tensor:
        """Get the diffusion time (t) from the log signal-to-noise ratio (lambda)."""
        if training:
            # SNR = -dist.icdf(t_trunc) => t = dist.cdf(-snr)
            loc = -2 * self.p_mean
            scale = 2 * self.p_std
            x = -log_snr_t
            t = 0.5 * (1 + ops.erf((x - loc) / (scale * math.sqrt(2.0))))
        else:  # sampling
            # SNR = -2 * rho * log(sigma_max ** (1/rho) + (1 - t) * (sigma_min ** (1/rho) - sigma_max ** (1/rho)))
            # => t = 1 - ((exp(-snr/(2*rho)) - sigma_max ** (1/rho)) / (sigma_min ** (1/rho) - sigma_max ** (1/rho)))
            t = 1 - (
                (ops.exp(-log_snr_t / (2 * self.rho)) - self.sigma_max ** (1 / self.rho))
                / (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
            )
        return t

    def derivative_log_snr(self, log_snr_t: Tensor, training: bool) -> Tensor:
        """Compute d/dt log(1 + e^(-snr(t))), which is used for the reverse SDE."""
        if training:
            raise NotImplementedError("Derivative of log SNR is not implemented for training mode.")
        # sampling mode
        t = self.get_t_from_log_snr(log_snr_t=log_snr_t, training=training)
        t_trunc = self._t_min + (self._t_max - self._t_min) * t

        # SNR = -2*rho*log(s_max + (1 - x)*(s_min - s_max))
        s_max = self.sigma_max ** (1 / self.rho)
        s_min = self.sigma_min ** (1 / self.rho)
        u = s_max + (1 - t_trunc) * (s_min - s_max)
        # d/dx snr = 2*rho*(s_min - s_max) / u
        dsnr_dx = 2 * self.rho * (s_min - s_max) / u

        # Using the chain rule on f(t) = log(1 + e^(-snr(t))):
        # f'(t) = - (e^{-snr(t)} / (1 + e^{-snr(t)})) * dsnr_dt
        dsnr_dt = dsnr_dx * (self._t_max - self._t_min)
        factor = ops.exp(-log_snr_t) / (1 + ops.exp(-log_snr_t))
        return -factor * dsnr_dt

    def get_weights_for_snr(self, log_snr_t: Tensor) -> Tensor:
        """Get weights for the signal-to-noise ratio (snr) for a given log signal-to-noise ratio (lambda)."""
        return ops.exp(-log_snr_t) + 0.5**2


@serializable
class DiffusionModel(InferenceNetwork):
    """Diffusion Model as described in this overview paper [1].

    [1] Variational Diffusion Models 2.0: Understanding Diffusion Model Objectives as the ELBO with Simple Data
        Augmentation: Kingma et al. (2023)
    [2] Score-Based Generative Modeling through Stochastic Differential Equations: Song et al. (2021)
    """

    MLP_DEFAULT_CONFIG = {
        "widths": (256, 256, 256, 256, 256),
        "activation": "mish",
        "kernel_initializer": "he_normal",
        "residual": True,
        "dropout": 0.0,
        "spectral_normalization": False,
    }

    INTEGRATE_DEFAULT_CONFIG = {
        "method": "euler",
        "steps": 100,
    }

    def __init__(
        self,
        subnet: str | type = "mlp",
        integrate_kwargs: dict[str, any] = None,
        subnet_kwargs: dict[str, any] = None,
        noise_schedule: str = "cosine",
        prediction_type: str = "v",
        **kwargs,
    ):
        """
        Initializes a diffusion model with configurable subnet architecture.

        This model learns a transformation from a Gaussian latent distribution to a target distribution using a
        specified subnet type, which can be an MLP or a custom network.

        The integration steps can be customized with additional parameters available in the respective
        configuration dictionary.

        Parameters
        ----------
        subnet : str or type, optional
            The architecture used for the transformation network. Can be "mlp" or a custom
            callable network. Default is "mlp".
        integrate_kwargs : dict[str, any], optional
            Additional keyword arguments for the integration process. Default is None.
        subnet_kwargs : dict[str, any], optional
            Keyword arguments passed to the subnet constructor or used to update the default MLP settings.
        noise_schedule : str, optional
            The noise schedule used for the diffusion process. Can be "linear", "cosine", or "edm".
            Default is "cosine".
        prediction_type: str, optional
            The type of prediction used in the diffusion model. Can be "eps", "v" or "F" (EDM). Default is "v".
        **kwargs
            Additional keyword arguments passed to the subnet and other components.
        """
        super().__init__(base_distribution="normal", **kwargs)

        if isinstance(noise_schedule, str):
            if noise_schedule == "linear":
                noise_schedule = LinearNoiseSchedule()
            elif noise_schedule == "cosine":
                noise_schedule = CosineNoiseSchedule()
            elif noise_schedule == "edm":
                noise_schedule = EDMNoiseSchedule()
            else:
                raise ValueError(f"Unknown noise schedule: {noise_schedule}")
        elif not isinstance(noise_schedule, NoiseSchedule):
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")
        self.noise_schedule = noise_schedule

        if prediction_type not in ["eps", "v", "F"]:  # F is EDM
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        self.prediction_type = prediction_type

        # clipping of prediction (after it was transformed to x-prediction)
        self._clip_min = -5.0
        self._clip_max = 5.0

        # latent distribution (not configurable)
        self.base_distribution = bf.distributions.DiagonalNormal(
            mean=0.0, std=self.noise_schedule.scale_base_distribution
        )
        self.integrate_kwargs = self.INTEGRATE_DEFAULT_CONFIG | (integrate_kwargs or {})
        self.seed_generator = keras.random.SeedGenerator()

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "mlp":
            subnet_kwargs = self.MLP_DEFAULT_CONFIG | subnet_kwargs

        self.subnet = find_network(subnet, **subnet_kwargs)
        self.output_projector = keras.layers.Dense(units=None, bias_initializer="zeros")

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        if self.built:
            return

        self.base_distribution.build(xz_shape)

        self.output_projector.units = xz_shape[-1]
        input_shape = list(xz_shape)

        # construct time vector
        input_shape[-1] += 1
        if conditions_shape is not None:
            input_shape[-1] += conditions_shape[-1]

        input_shape = tuple(input_shape)

        self.subnet.build(input_shape)
        out_shape = self.subnet.compute_output_shape(input_shape)
        self.output_projector.build(out_shape)

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "subnet": self.subnet,
            "noise_schedule": self.noise_schedule,
            "integrate_kwargs": self.integrate_kwargs,
            "prediction_type": self.prediction_type,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def convert_prediction_to_x(
        self, pred: Tensor, z: Tensor, alpha_t: Tensor, sigma_t: Tensor, log_snr_t: Tensor, clip_x: bool
    ) -> Tensor:
        """Convert the prediction of the neural network to the x space."""
        if self.prediction_type == "v":
            # convert v into x
            x = alpha_t * z - sigma_t * pred
        elif self.prediction_type == "e":
            # convert noise prediction into x
            x = (z - sigma_t * pred) / alpha_t
        elif self.prediction_type == "x":
            x = pred
        elif self.prediction_type == "score":
            x = (z + sigma_t**2 * pred) / alpha_t
        else:  # self.prediction_type == 'F':  # EDM
            sigma_data = self.noise_schedule.sigma_data
            x1 = (sigma_data**2 * alpha_t) / (ops.exp(-log_snr_t) + sigma_data**2)
            x2 = ops.exp(-log_snr_t / 2) * sigma_data / ops.sqrt(ops.exp(-log_snr_t) + sigma_data**2)
            x = x1 * z + x2 * pred

        if clip_x:
            x = keras.ops.clip(x, self._clip_min, self._clip_max)
        return x

    def velocity(
        self,
        xz: Tensor,
        time: float | Tensor,
        conditions: Tensor = None,
        training: bool = False,
        clip_x: bool = True,
    ) -> Tensor:
        # calculate the current noise level and transform into correct shape
        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = keras.ops.broadcast_to(log_snr_t, keras.ops.shape(xz)[:-1] + (1,))
        alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t, training=training)

        if conditions is None:
            xtc = keras.ops.concatenate([xz, log_snr_t], axis=-1)
        else:
            xtc = keras.ops.concatenate([xz, log_snr_t, conditions], axis=-1)
        pred = self.output_projector(self.subnet(xtc, training=training), training=training)

        x_pred = self.convert_prediction_to_x(
            pred=pred, z=xz, alpha_t=alpha_t, sigma_t=sigma_t, log_snr_t=log_snr_t, clip_x=clip_x
        )
        # convert x to score
        score = (alpha_t * x_pred - xz) / ops.square(sigma_t)

        # compute velocity for the ODE depending on the noise schedule
        f, g = self.noise_schedule.get_drift_diffusion(log_snr_t=log_snr_t, x=xz)
        out = f - 0.5 * ops.square(g) * score

        # todo: for the SDE: d(z) = [ f(z, t) - g(t)^2 * score(z, lambda) ] dt + g(t) dW
        return out

    def _velocity_trace(
        self,
        xz: Tensor,
        time: Tensor,
        conditions: Tensor = None,
        max_steps: int = None,
        training: bool = False,
    ) -> (Tensor, Tensor):
        def f(x):
            return self.velocity(x, time=time, conditions=conditions, training=training)

        v, trace = jacobian_trace(f, xz, max_steps=max_steps, seed=self.seed_generator, return_output=True)

        return v, keras.ops.expand_dims(trace, axis=-1)

    def _forward(
        self,
        x: Tensor,
        conditions: Tensor = None,
        density: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        integrate_kwargs = (
            {
                "start_time": self.noise_schedule._t_min,
                "stop_time": self.noise_schedule._t_max,
            }
            | self.integrate_kwargs
            | kwargs
        )
        if density:

            def deltas(time, xz):
                v, trace = self._velocity_trace(xz, time=time, conditions=conditions, training=training)
                return {"xz": v, "trace": trace}

            state = {
                "xz": x,
                "trace": keras.ops.zeros(keras.ops.shape(x)[:-1] + (1,), dtype=keras.ops.dtype(x)),
            }
            state = integrate(
                deltas,
                state,
                **integrate_kwargs,
            )

            z = state["xz"]
            log_density = self.base_distribution.log_prob(z) + keras.ops.squeeze(state["trace"], axis=-1)

            return z, log_density

        def deltas(time, xz):
            return {"xz": self.velocity(xz, time=time, conditions=conditions, training=training)}

        state = {"xz": x}
        state = integrate(
            deltas,
            state,
            **integrate_kwargs,
        )
        z = state["xz"]
        return z

    def _inverse(
        self,
        z: Tensor,
        conditions: Tensor = None,
        density: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        integrate_kwargs = (
            {
                "start_time": self.noise_schedule._t_max,
                "stop_time": self.noise_schedule._t_min,
            }
            | self.integrate_kwargs
            | kwargs
        )
        if density:

            def deltas(time, xz):
                v, trace = self._velocity_trace(xz, time=time, conditions=conditions, training=training)
                return {"xz": v, "trace": trace}

            state = {
                "xz": z,
                "trace": keras.ops.zeros(keras.ops.shape(z)[:-1] + (1,), dtype=keras.ops.dtype(z)),
            }
            state = integrate(deltas, state, **integrate_kwargs)

            x = state["xz"]
            log_density = self.base_distribution.log_prob(z) - keras.ops.squeeze(state["trace"], axis=-1)

            return x, log_density

        def deltas(time, xz):
            return {"xz": self.velocity(xz, time=time, conditions=conditions, training=training)}

        state = {"xz": z}
        state = integrate(
            deltas,
            state,
            **integrate_kwargs,
        )

        x = state["xz"]
        return x

    def compute_metrics(
        self,
        x: Tensor | Sequence[Tensor, ...],
        conditions: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:
        training = stage == "training"
        if not self.built:
            xz_shape = keras.ops.shape(x)
            conditions_shape = None if conditions is None else keras.ops.shape(conditions)
            self.build(xz_shape, conditions_shape)

        # sample training diffusion time as low discrepancy sequence to decrease variance
        # t_i = \mod (u_0 + i/k, 1)
        u0 = keras.random.uniform(shape=(1,))
        i = ops.arange(0, keras.ops.shape(x)[0])  # tensor of indices
        t = (u0 + i / keras.ops.shape(x)[0]) % 1
        # i = keras.random.randint((keras.ops.shape(x)[0],), minval=0, maxval=self._timesteps)
        # t = keras.ops.cast(i, keras.ops.dtype(x)) / keras.ops.cast(self._timesteps, keras.ops.dtype(x))

        # calculate the noise level
        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t, training=training), x)
        alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t, training=training)

        # generate noise vector
        eps_t = keras.random.normal(ops.shape(x), dtype=ops.dtype(x), seed=self.seed_generator)

        # diffuse x
        diffused_x = alpha_t * x + sigma_t * eps_t

        # calculate output of the network
        if conditions is None:
            xtc = keras.ops.concatenate([diffused_x, log_snr_t], axis=-1)
        else:
            xtc = keras.ops.concatenate([diffused_x, log_snr_t, conditions], axis=-1)
        pred = self.output_projector(self.subnet(xtc, training=training), training=training)

        x_pred = self.convert_prediction_to_x(
            pred=pred, z=diffused_x, alpha_t=alpha_t, sigma_t=sigma_t, log_snr_t=log_snr_t, clip_x=True
        )
        # convert x to epsilon prediction
        out = (alpha_t * diffused_x - x_pred) / sigma_t

        # Calculate loss based on noise prediction
        weights_for_snr = self.noise_schedule.get_weights_for_snr(log_snr_t=log_snr_t)
        loss = weights_for_snr * ops.mean((out - eps_t) ** 2, axis=-1)

        # apply sample weight
        loss = weighted_mean(loss, sample_weight)

        base_metrics = super().compute_metrics(x, conditions, sample_weight, stage)
        return base_metrics | {"loss": loss}
