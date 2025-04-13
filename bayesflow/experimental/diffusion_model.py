from collections.abc import Sequence
import keras
from keras import ops
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor, Shape
import bayesflow as bf
from bayesflow.networks import InferenceNetwork

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
    """Diffusion Model as described as Elucidated Diffusion Model in [1].

    [1] Elucidating the Design Space of Diffusion-Based Generative Models: arXiv:2206.00364
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

        # internal tunable parameters not intended to be modified by the average user
        self.max_sigma = kwargs.get("max_sigma", 80.0)
        self.min_sigma = kwargs.get("min_sigma", 1e-4)
        self.rho = kwargs.get("rho", 7)
        # hyper-parameters for sampling the noise level
        self.p_mean = kwargs.get("p_mean", -1.2)
        self.p_std = kwargs.get("p_std", 1.2)

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
        return 0.25 * ops.log(sigma)

    def _denoiser_fn(
        self,
        xz: Tensor,
        sigma: Tensor,
        conditions: Tensor = None,
        training: bool = False,
    ):
        # calculate output of the network
        c_in = self._c_in_fn(sigma)
        c_noise = self._c_noise_fn(sigma)
        xz_pre = c_in * xz
        if conditions is None:
            xtc = keras.ops.concatenate([xz_pre, c_noise], axis=-1)
        else:
            xtc = keras.ops.concatenate([xz_pre, c_noise, conditions], axis=-1)
        out = self.output_projector(self.subnet(xtc, training=training), training=training)
        return self._c_skip_fn(sigma) * xz + self._c_out_fn(sigma) * out

    def velocity(
        self,
        xz: Tensor,
        sigma: float | Tensor,
        conditions: Tensor = None,
        training: bool = False,
    ) -> Tensor:
        # transform sigma vector into correct shape
        sigma = keras.ops.convert_to_tensor(sigma, dtype=keras.ops.dtype(xz))
        sigma = expand_right_as(sigma, xz)
        sigma = keras.ops.broadcast_to(sigma, keras.ops.shape(xz)[:-1] + (1,))

        d = self._denoiser_fn(xz, sigma, conditions, training=training)
        return (xz - d) / sigma

    def _velocity_trace(
        self,
        xz: Tensor,
        sigma: Tensor,
        conditions: Tensor = None,
        max_steps: int = None,
        training: bool = False,
    ) -> (Tensor, Tensor):
        def f(x):
            return self.velocity(x, sigma=sigma, conditions=conditions, training=training)

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
                v, trace = self._velocity_trace(xz, sigma=time, conditions=conditions, training=training)
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
            return {"xz": self.velocity(xz, sigma=time, conditions=conditions, training=training)}

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
                v, trace = self._velocity_trace(xz, sigma=time, conditions=conditions, training=training)
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
            return {"xz": self.velocity(xz, sigma=time, conditions=conditions, training=training)}

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

        # sample log-noise level
        log_sigma = self.p_mean + self.p_std * keras.random.normal(
            ops.shape(x)[:1], dtype=ops.dtype(x), seed=self.seed_generator
        )
        # noise level with shape (batch_size, 1)
        sigma = ops.exp(log_sigma)[:, None]

        # generate noise vector
        z = sigma * keras.random.normal(ops.shape(x), dtype=ops.dtype(x), seed=self.seed_generator)

        # calculate preconditioning
        c_skip = self._c_skip_fn(sigma)
        c_out = self._c_out_fn(sigma)
        c_in = self._c_in_fn(sigma)
        c_noise = self._c_noise_fn(sigma)
        xz_pre = c_in * (x + z)

        # calculate output of the network
        if conditions is None:
            xtc = keras.ops.concatenate([xz_pre, c_noise], axis=-1)
        else:
            xtc = keras.ops.concatenate([xz_pre, c_noise, conditions], axis=-1)

        out = self.output_projector(self.subnet(xtc, training=training), training=training)

        # Calculate loss:
        lam = 1 / c_out[:, 0] ** 2
        effective_weight = lam * c_out[:, 0] ** 2
        unweighted_loss = ops.mean((out - 1 / c_out * (x - c_skip * (x + z))) ** 2, axis=-1)
        loss = effective_weight * unweighted_loss
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
