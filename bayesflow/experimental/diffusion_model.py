from collections.abc import Sequence
import keras
from keras import ops
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor, Shape
import bayesflow as bf
from bayesflow.networks import InferenceNetwork
import math

from bayesflow.utils import (
    expand_right_as,
    find_network,
    jacobian_trace,
    keras_kwargs,
    serialize_value_or_type,
    deserialize_value_or_type,
    weighted_mean,
    integrate,
)


@serializable(package="bayesflow.networks")
class DiffusionModel(InferenceNetwork):
    """Diffusion Model as described in this overview paper [1].

    [1] Variational Diffusion Models 2.0: Understanding Diffusion Model Objectives as the ELBO with Simple Data
        Augmentation: Kingma et al. (2023)
    [2] Score-Based Generative Modeling through Stochastic Differential Equations: Song et al. (2021)
    [3] Elucidating the Design Space of Diffusion-Based Generative Models: arXiv:2206.00364

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
        sigma_data=1.0,
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
        sigma_data : float, optional
            Averaged standard deviation of the target distribution. Default is 1.0.
        **kwargs
            Additional keyword arguments passed to the subnet and other components.
        """

        super().__init__(base_distribution=None, **keras_kwargs(kwargs))

        # todo: clean up these configurations
        # EDM hyper-parameters
        # internal tunable parameters not intended to be modified by the average user
        self.max_sigma = kwargs.get("max_sigma", 80.0)
        self.min_sigma = kwargs.get("min_sigma", 1e-4)
        self.rho = kwargs.get("rho", 7)
        # hyper-parameters for sampling the noise level
        self.p_mean = kwargs.get("p_mean", -1.2)
        self.p_std = kwargs.get("p_std", 1.2)
        self._noise_schedule = kwargs.get("noise_schedule", "EDM")

        # general hyper-parameters
        self._train_time = kwargs.get("train_time", "continuous")
        self._timesteps = kwargs.get("timesteps", None)
        if self._train_time == "discrete":
            if not isinstance(self._timesteps, int):
                raise ValueError('timesteps must be defined, if "discrete" training time is set')
        self._loss_type = kwargs.get("loss_type", "eps")
        self._weighting_function = kwargs.get("weighting_function", None)
        self._log_snr_min = kwargs.get("log_snr_min", -15)
        self._log_snr_max = kwargs.get("log_snr_max", 15)
        self._t_min = self._get_t_from_log_snr(log_snr_t=self._log_snr_max)
        self._t_max = self._get_t_from_log_snr(log_snr_t=self._log_snr_min)
        self._s_shift_cosine = kwargs.get("s_shift_cosine", 0.0)

        # latent distribution (not configurable)
        self.base_distribution = bf.distributions.DiagonalNormal(mean=0.0, std=self.max_sigma)

        self.integrate_kwargs = self.INTEGRATE_DEFAULT_CONFIG | (integrate_kwargs or {})

        self.sigma_data = sigma_data

        self.seed_generator = keras.random.SeedGenerator()

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "mlp":
            subnet_kwargs = self.MLP_DEFAULT_CONFIG | subnet_kwargs

        self.subnet = find_network(subnet, **subnet_kwargs)
        self.output_projector = keras.layers.Dense(units=None, bias_initializer="zeros")

        # serialization: store all parameters necessary to call __init__
        self.config = {
            "integrate_kwargs": self.integrate_kwargs,
            "subnet_kwargs": subnet_kwargs,
            "sigma_data": sigma_data,
            **kwargs,
        }
        self.config = serialize_value_or_type(self.config, "subnet", subnet)

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        super().build(xz_shape, conditions_shape=conditions_shape)

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
        return base_config | self.config

    @classmethod
    def from_config(cls, config):
        config = deserialize_value_or_type(config, "subnet")
        return cls(**config)

    def _c_skip_fn(self, sigma):
        return self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

    def _c_out_fn(self, sigma):
        return sigma * self.sigma_data / ops.sqrt(self.sigma_data**2 + sigma**2)

    def _c_in_fn(self, sigma):
        return 1.0 / ops.sqrt(sigma**2 + self.sigma_data**2)

    def _c_noise_fn(self, sigma):
        return 0.25 * ops.log(sigma)  # this is the snr times a constant

    def velocity(
        self,
        xz: Tensor,
        time: float | Tensor,
        conditions: Tensor = None,
        training: bool = False,
        clip_x: bool = True,
    ) -> Tensor:
        # calculate the current noise level and transform into correct shape
        log_snr_t = expand_right_as(self._get_log_snr(t=time), xz)
        alpha_t, sigma_t = self._get_alpha_sigma(log_snr_t=log_snr_t)

        if self._noise_schedule == "EDM":
            # scale the input
            xz = alpha_t * xz

        if conditions is None:
            xtc = keras.ops.concatenate([xz, log_snr_t], axis=-1)
        else:
            xtc = keras.ops.concatenate([xz, log_snr_t, conditions], axis=-1)
        pred = self.output_projector(self.subnet(xtc, training=training), training=training)

        if self._noise_schedule == "EDM":
            # scale the output
            s = ops.exp(-1 / 2 * log_snr_t)
            pred_scaled = self._c_skip_fn(s) * xz + self._c_out_fn(s) * pred
            out = (xz - pred_scaled) / s
        else:
            # first convert prediction to x-prediction
            if self._loss_type == "eps":
                x_pred = (xz - sigma_t * pred) / alpha_t
            else:  # self._loss_type == 'v':
                x_pred = alpha_t * xz - sigma_t * pred

            # clip x if necessary
            if clip_x:
                x_pred = ops.clip(x_pred, -5, 5)
            # convert x to score
            score = (alpha_t * x_pred - xz) / ops.square(sigma_t)
            # compute velocity for the ODE depending on the noise schedule
            f, g = self._get_drift_diffusion(log_snr_t=log_snr_t, x=xz)
            out = f - 0.5 * ops.square(g) * score
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
        integrate_kwargs = self.integrate_kwargs | kwargs
        if isinstance(integrate_kwargs["steps"], int):
            # set schedule for specified number of steps
            integrate_kwargs["steps"] = self._integration_schedule(integrate_kwargs["steps"], dtype=ops.dtype(x))
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
        integrate_kwargs = self.integrate_kwargs | kwargs
        if isinstance(integrate_kwargs["steps"], int):
            # set schedule for specified number of steps
            integrate_kwargs["steps"] = self._integration_schedule(
                integrate_kwargs["steps"], inverse=True, dtype=ops.dtype(z)
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

    def _get_drift_diffusion(self, log_snr_t, x=None):  # t is not truncated
        """
        Compute d/dt log(1 + e^(-snr(t))) for the truncated schedules.
        """
        t = self._get_t_from_log_snr(log_snr_t=log_snr_t)
        # Compute the truncated time t_trunc
        t_trunc = self._t_min + (self._t_max - self._t_min) * t

        # Compute d/dx snr(x) based on the noise schedule
        if self._noise_schedule == "linear":
            # d/dx snr(x) = - 2*x*exp(x^2) / (exp(x^2) - 1)
            dsnr_dx = -(2 * t_trunc * ops.exp(t_trunc**2)) / (ops.exp(t_trunc**2) - 1)
        elif self._noise_schedule == "cosine":
            # d/dx snr(x) = -2*pi/sin(pi*x)
            dsnr_dx = -(2 * math.pi) / ops.sin(math.pi * t_trunc)
        elif self._noise_schedule == "flow_matching":
            # d/dx snr(x) = -2/(x*(1-x))
            dsnr_dx = -2 / (t_trunc * (1 - t_trunc))
        else:
            raise ValueError("Invalid 'noise_schedule'.")

        # Chain rule: d/dt snr(t) = d/dx snr(x) * (t_max - t_min)
        dsnr_dt = dsnr_dx * (self._t_max - self._t_min)

        # Using the chain rule on f(t) = log(1 + e^(-snr(t))):
        # f'(t) = - (e^{-snr(t)} / (1 + e^{-snr(t)})) * dsnr_dt
        factor = ops.exp(-log_snr_t) / (1 + ops.exp(-log_snr_t))

        beta_t = -factor * dsnr_dt
        g = ops.sqrt(beta_t)  # diffusion term
        if x is None:
            return g
        f = -0.5 * beta_t * x  # drift term
        return f, g

    def _get_log_snr(self, t: Tensor) -> Tensor:
        """get the log signal-to-noise ratio (lambda) for a given diffusion time"""
        if self._noise_schedule == "EDM":
            # EDM defines tilde sigma ~ N(p_mean, p_std^2)
            # tilde sigma^2 = exp(-lambda), hence lambda = -2 * log(sigma)
            # sample noise
            log_sigma_tilde = self.p_mean + self.p_std * keras.random.normal(
                ops.shape(t), dtype=ops.dtype(t), seed=self.seed_generator
            )
            # calculate the log signal-to-noise ratio
            log_snr_t = -2 * log_sigma_tilde
            return log_snr_t

        t_trunc = self._t_min + (self._t_max - self._t_min) * t
        if self._noise_schedule == "linear":
            log_snr_t = -ops.log(ops.exp(ops.square(t_trunc)) - 1)
        elif self._noise_schedule == "cosine":  # this is usually used with variance_preserving
            log_snr_t = -2 * ops.log(ops.tan(math.pi * t_trunc / 2)) + 2 * self._s_shift_cosine
        elif self._noise_schedule == "flow_matching":  # this usually used with sub_variance_preserving
            log_snr_t = 2 * ops.log((1 - t_trunc) / t_trunc)
        else:
            raise ValueError("Unknown noise schedule: {}".format(self._noise_schedule))
        return log_snr_t

    def _get_t_from_log_snr(self, log_snr_t) -> Tensor:
        # Invert the noise scheduling to recover t (not truncated)
        if self._noise_schedule == "linear":
            # SNR = -log(exp(t^2) - 1)
            # => t = sqrt(log(1 + exp(-snr)))
            t = ops.sqrt(ops.log(1 + ops.exp(-log_snr_t)))
        elif self._noise_schedule == "cosine":
            # SNR = -2 * log(tan(pi*t/2))
            # => t = 2/pi * arctan(exp(-snr/2))
            t = 2 / math.pi * ops.arctan(ops.exp((2 * self._s_shift_cosine - log_snr_t) / 2))
        elif self._noise_schedule == "flow_matching":
            # SNR = 2 * log((1-t)/t)
            # => t = 1 / (1 + exp(snr/2))
            t = 1 / (1 + ops.exp(log_snr_t / 2))
        elif self._noise_schedule == "EDM":
            raise NotImplementedError
        else:
            raise ValueError("Unknown noise schedule: {}".format(self._noise_schedule))
        return t

    def _get_alpha_sigma(self, log_snr_t: Tensor) -> tuple[Tensor, Tensor]:
        if self._noise_schedule == "EDM":
            # EDM: noisy_x = c_in * (x + s * e) = c_in * x + c_in * s * e
            # s^2 = exp(-lambda)
            s = ops.exp(-1 / 2 * log_snr_t)
            c_in = self._c_in_fn(s)

            # alpha = c_in(s), sigma = c_in * s
            alpha_t = c_in
            sigma_t = c_in * s
        else:
            # variance preserving noise schedules
            alpha_t = keras.ops.sqrt(keras.ops.sigmoid(log_snr_t))
            sigma_t = keras.ops.sqrt(keras.ops.sigmoid(-log_snr_t))
        return alpha_t, sigma_t

    def _get_weights_for_snr(self, log_snr_t: Tensor) -> Tensor:
        if self._noise_schedule == "EDM":
            # EDM: weights are constructed elsewhere
            weights = ops.ones_like(log_snr_t)
            return weights

        if self._weighting_function == "likelihood_weighting":  # based on Song et al. (2021)
            g_t = self._get_drift_diffusion(log_snr_t=log_snr_t)
            sigma_t = self._get_alpha_sigma(log_snr_t=log_snr_t)[1]
            weights = ops.square(g_t / sigma_t)
        elif self._weighting_function == "sigmoid":  # based on Kingma et al. (2023)
            weights = ops.sigmoid(-log_snr_t / 2)
        elif self._weighting_function == "min-snr":  # based on Hang et al. (2023)
            gamma = 5
            weights = 1 / ops.cosh(log_snr_t / 2) * ops.minimum(ops.ones_like(log_snr_t), gamma * ops.exp(-log_snr_t))
        else:
            weights = ops.ones_like(log_snr_t)
        return weights

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

        # sample training diffusion time
        if self._train_time == "continuous":
            t = keras.random.uniform((keras.ops.shape(x)[0],))
        elif self._train_time == "discrete":
            i = keras.random.randint((keras.ops.shape(x)[0],), minval=0, maxval=self._timesteps)
            t = keras.ops.cast(i, keras.ops.dtype(x)) / keras.ops.cast(self._timesteps, keras.ops.dtype(x))
        else:
            raise NotImplementedError(f"Training time {self._train_time} not implemented")

        # calculate the noise level
        log_snr_t = expand_right_as(self._get_log_snr(t), x)
        alpha_t, sigma_t = self._get_alpha_sigma(log_snr_t=log_snr_t)

        # generate noise vector
        eps_t = keras.random.normal(ops.shape(x), dtype=ops.dtype(x), seed=self.seed_generator)

        # diffuse x
        diffused_x = alpha_t * x + sigma_t * eps_t

        # calculate output of the network
        if conditions is None:
            xtc = keras.ops.concatenate([diffused_x, log_snr_t], axis=-1)
        else:
            xtc = keras.ops.concatenate([diffused_x, log_snr_t, conditions], axis=-1)

        out = self.output_projector(self.subnet(xtc, training=training), training=training)

        # Calculate loss
        weights_for_snr = self._get_weights_for_snr(log_snr_t=log_snr_t)
        if self._loss_type == "eps":
            loss = weights_for_snr * ops.mean((out - eps_t) ** 2, axis=-1)
        elif self._loss_type == "v":
            v_t = alpha_t * eps_t - sigma_t * x
            loss = weights_for_snr * ops.mean((out - v_t) ** 2, axis=-1)
        elif self._loss_type == "EDM":
            s = ops.exp(-1 / 2 * log_snr_t)
            c_skip = self._c_skip_fn(s)
            c_out = self._c_out_fn(s)
            lam = 1 / c_out[:, 0] ** 2
            effective_weight = lam * c_out[:, 0] ** 2
            unweighted_loss = ops.mean((out - 1 / c_out * (x - c_skip * (x + s + eps_t))) ** 2, axis=-1)
            loss = effective_weight * unweighted_loss
        else:
            raise ValueError(f"Unknown loss type: {self._loss_type}")

        loss = weighted_mean(loss, sample_weight)

        base_metrics = super().compute_metrics(x, conditions, sample_weight, stage)
        return base_metrics | {"loss": loss}

    def _integration_schedule(self, steps, inverse=False, dtype=None):
        def sigma_i(i, steps):
            N = steps + 1
            return (
                self.max_sigma ** (1 / self.rho)
                + (i / (N - 1)) * (self.min_sigma ** (1 / self.rho) - self.max_sigma ** (1 / self.rho))
            ) ** self.rho

        steps = sigma_i(ops.arange(steps + 1, dtype=dtype), steps)
        if not inverse:
            steps = ops.flip(steps)
        return steps
