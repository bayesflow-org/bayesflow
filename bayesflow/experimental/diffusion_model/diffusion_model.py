from collections.abc import Sequence
from typing import Literal

import keras
from keras import ops

from bayesflow.networks import InferenceNetwork
from bayesflow.types import Tensor, Shape
from bayesflow.utils import (
    expand_right_as,
    find_network,
    jacobian_trace,
    layer_kwargs,
    weighted_mean,
    integrate,
    integrate_stochastic,
    logging,
    tensor_utils,
)
from .dispatch import find_noise_schedule
from bayesflow.utils.serialization import serialize, deserialize, serializable


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
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
        "method": "euler",  # or euler_maruyama
        "steps": 250,
    }

    def __init__(
        self,
        *,
        subnet: str | type = "mlp",
        integrate_kwargs: dict[str, any] = None,
        noise_schedule: Literal["edm", "cosine"] | dict | type = "edm",
        prediction_type: Literal["velocity", "noise", "F"] = "F",
        **kwargs,
    ):
        """
        Initializes a diffusion model with configurable subnet architecture.

        This model learns a transformation from a Gaussian latent distribution to a target distribution using a
        specified subnet type, which can be an MLP or a custom network.

        The integration can be customized with additional parameters available in the integrate_kwargs
        configuration dictionary. Different noise schedules and prediction types are available.

        Parameters
        ----------
        subnet : str or type, optional
            The architecture used for the transformation network. Can be "mlp" or a custom
            callable network. Default is "mlp".
        integrate_kwargs : dict[str, any], optional
            Additional keyword arguments for the integration process. Default is None.
        noise_schedule : Literal['edm', 'cosine'], dict or type, optional
            The noise schedule used for the diffusion process. Can be "cosine" or "edm" or a custom noise schedule.
            You can also pass a dictionary with the configuration for the noise schedule, e.g.,
                {'type': cosine, 's_shift_cosine': 1.0}
            Default is "edm".
        prediction_type: Literal['velocity', 'noise', 'F'], optional
            The type of prediction used in the diffusion model. Can be "velocity", "noise" or "F" (EDM).
             Default is "F".
        **kwargs
            Additional keyword arguments passed to the subnet and other components.
        """
        super().__init__(base_distribution="normal", **kwargs)

        self.noise_schedule = find_noise_schedule(noise_schedule)
        self.noise_schedule.validate()

        if prediction_type not in ["noise", "velocity", "F"]:  # F is EDM
            raise ValueError(f"Unknown prediction type: {prediction_type}")
        self._prediction_type = prediction_type
        self._loss_type = kwargs.get("loss_type", "noise")
        if self._loss_type not in ["noise", "velocity", "F"]:
            raise ValueError(f"Unknown loss type: {self._loss_type}")
        if self._loss_type != "noise":
            logging.warning(
                "the standard schedules have weighting functions defined for the noise prediction loss. "
                "You might want to replace them, if you use a different loss function."
            )

        # clipping of prediction (after it was transformed to x-prediction)
        # keeping this private for now, as it is usually not required in SBI and somewhat dangerous
        self._clip_x = kwargs.get("clip_x", None)
        if self._clip_x is not None:
            if len(self._clip_x) != 2 or self._clip_x[0] > self._clip_x[1]:
                raise ValueError("'clip_x' has to be a list or tuple with the values [x_min, x_max]")

        self.integrate_kwargs = self.INTEGRATE_DEFAULT_CONFIG | (integrate_kwargs or {})
        self.seed_generator = keras.random.SeedGenerator()

        if subnet == "mlp":
            self.subnet = find_network(subnet, **self.MLP_DEFAULT_CONFIG)
        else:
            self.subnet = find_network(subnet)
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
            "prediction_type": self._prediction_type,
            "loss_type": self._loss_type,
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def convert_prediction_to_x(
        self, pred: Tensor, z: Tensor, alpha_t: Tensor, sigma_t: Tensor, log_snr_t: Tensor
    ) -> Tensor:
        """Convert the prediction of the neural network to the x space."""
        if self._prediction_type == "velocity":
            # convert v into x
            x = alpha_t * z - sigma_t * pred
        elif self._prediction_type == "noise":
            # convert noise prediction into x
            x = (z - sigma_t * pred) / alpha_t
        elif self._prediction_type == "F":  # EDM
            sigma_data = self.noise_schedule.sigma_data if hasattr(self.noise_schedule, "sigma_data") else 1.0
            x1 = (sigma_data**2 * alpha_t) / (ops.exp(-log_snr_t) + sigma_data**2)
            x2 = ops.exp(-log_snr_t / 2) * sigma_data / ops.sqrt(ops.exp(-log_snr_t) + sigma_data**2)
            x = x1 * z + x2 * pred
        elif self._prediction_type == "x":
            x = pred
        else:  # "score"
            x = (z + sigma_t**2 * pred) / alpha_t

        if self._clip_x is not None:
            x = ops.clip(x, self._clip_x[0], self._clip_x[1])
        return x

    def velocity(
        self,
        xz: Tensor,
        time: float | Tensor,
        stochastic_solver: bool,
        conditions: Tensor = None,
        training: bool = False,
    ) -> Tensor:
        # calculate the current noise level and transform into correct shape
        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))
        alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t, training=training)

        if conditions is None:
            xtc = tensor_utils.concatenate_valid([xz, self._transform_log_snr(log_snr_t)], axis=-1)
        else:
            xtc = tensor_utils.concatenate_valid([xz, self._transform_log_snr(log_snr_t), conditions], axis=-1)
        pred = self.output_projector(self.subnet(xtc, training=training), training=training)

        x_pred = self.convert_prediction_to_x(pred=pred, z=xz, alpha_t=alpha_t, sigma_t=sigma_t, log_snr_t=log_snr_t)
        # convert x to score
        score = (alpha_t * x_pred - xz) / ops.square(sigma_t)

        # compute velocity f, g of the SDE or ODE
        f, g_squared = self.noise_schedule.get_drift_diffusion(log_snr_t=log_snr_t, x=xz)

        if stochastic_solver:
            # for the SDE: d(z) = [f(z, t) - g(t) ^ 2 * score(z, lambda )] dt + g(t) dW
            out = f - g_squared * score
        else:
            # for the ODE: d(z) = [f(z, t) - 0.5 * g(t) ^ 2 * score(z, lambda )] dt
            out = f - 0.5 * g_squared * score

        return out

    def compute_diffusion_term(
        self,
        xz: Tensor,
        time: float | Tensor,
        training: bool = False,
    ) -> Tensor:
        # calculate the current noise level and transform into correct shape
        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))
        g_squared = self.noise_schedule.get_drift_diffusion(log_snr_t=log_snr_t)
        return ops.sqrt(g_squared)

    def _velocity_trace(
        self,
        xz: Tensor,
        time: Tensor,
        conditions: Tensor = None,
        max_steps: int = None,
        training: bool = False,
    ) -> (Tensor, Tensor):
        def f(x):
            return self.velocity(x, time=time, stochastic_solver=False, conditions=conditions, training=training)

        v, trace = jacobian_trace(f, xz, max_steps=max_steps, seed=self.seed_generator, return_output=True)

        return v, ops.expand_dims(trace, axis=-1)

    def _transform_log_snr(self, log_snr: Tensor) -> Tensor:
        """Transform the log_snr to the range [-1, 1] for the diffusion process."""
        log_snr_min = self.noise_schedule.log_snr_min
        log_snr_max = self.noise_schedule.log_snr_max

        # Calculate normalized value within the range [0, 1]
        normalized_snr = (log_snr - log_snr_min) / (log_snr_max - log_snr_min)

        # Scale to [-1, 1] range
        scaled_value = 2 * normalized_snr - 1
        return scaled_value

    def _forward(
        self,
        x: Tensor,
        conditions: Tensor = None,
        density: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        integrate_kwargs = {"start_time": 0.0, "stop_time": 1.0}
        integrate_kwargs = integrate_kwargs | self.integrate_kwargs
        integrate_kwargs = integrate_kwargs | kwargs
        if integrate_kwargs["method"] == "euler_maruyama":
            raise ValueError("Stochastic methods are not supported for forward integration.")

        if density:

            def deltas(time, xz):
                v, trace = self._velocity_trace(xz, time=time, conditions=conditions, training=training)
                return {"xz": v, "trace": trace}

            state = {
                "xz": x,
                "trace": ops.zeros(ops.shape(x)[:-1] + (1,), dtype=ops.dtype(x)),
            }
            state = integrate(
                deltas,
                state,
                **integrate_kwargs,
            )

            z = state["xz"]
            log_density = self.base_distribution.log_prob(z) + ops.squeeze(state["trace"], axis=-1)

            return z, log_density

        def deltas(time, xz):
            return {
                "xz": self.velocity(xz, time=time, stochastic_solver=False, conditions=conditions, training=training)
            }

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
        integrate_kwargs = {"start_time": 1.0, "stop_time": 0.0}
        integrate_kwargs = integrate_kwargs | self.integrate_kwargs
        integrate_kwargs = integrate_kwargs | kwargs
        if density:
            if integrate_kwargs["method"] == "euler_maruyama":
                raise ValueError("Stochastic methods are not supported for density computation.")

            def deltas(time, xz):
                v, trace = self._velocity_trace(xz, time=time, conditions=conditions, training=training)
                return {"xz": v, "trace": trace}

            state = {
                "xz": z,
                "trace": ops.zeros(ops.shape(z)[:-1] + (1,), dtype=ops.dtype(z)),
            }
            state = integrate(deltas, state, **integrate_kwargs)

            x = state["xz"]
            log_density = self.base_distribution.log_prob(z) - ops.squeeze(state["trace"], axis=-1)

            return x, log_density

        state = {"xz": z}
        if integrate_kwargs["method"] == "euler_maruyama":

            def deltas(time, xz):
                return {
                    "xz": self.velocity(xz, time=time, stochastic_solver=True, conditions=conditions, training=training)
                }

            def diffusion(time, xz):
                return {"xz": self.compute_diffusion_term(xz, time=time, training=training)}

            state = integrate_stochastic(
                drift_fn=deltas,
                diffusion_fn=diffusion,
                state=state,
                seed=self.seed_generator,
                **integrate_kwargs,
            )
        else:

            def deltas(time, xz):
                return {
                    "xz": self.velocity(
                        xz, time=time, stochastic_solver=False, conditions=conditions, training=training
                    )
                }

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
        # use same noise schedule for training and validation to keep them comparable
        noise_schedule_training_stage = stage == "training" or stage == "validation"
        if not self.built:
            xz_shape = ops.shape(x)
            conditions_shape = None if conditions is None else ops.shape(conditions)
            self.build(xz_shape, conditions_shape)

        # sample training diffusion time as low discrepancy sequence to decrease variance
        u0 = keras.random.uniform(shape=(1,), dtype=ops.dtype(x), seed=self.seed_generator)
        i = ops.arange(0, ops.shape(x)[0], dtype=ops.dtype(x))  # tensor of indices
        t = (u0 + i / ops.cast(ops.shape(x)[0], dtype=ops.dtype(x))) % 1

        # calculate the noise level
        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t, training=noise_schedule_training_stage), x)
        alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(
            log_snr_t=log_snr_t, training=noise_schedule_training_stage
        )
        weights_for_snr = self.noise_schedule.get_weights_for_snr(log_snr_t=log_snr_t)

        # generate noise vector
        eps_t = keras.random.normal(ops.shape(x), dtype=ops.dtype(x), seed=self.seed_generator)

        # diffuse x
        diffused_x = alpha_t * x + sigma_t * eps_t

        # calculate output of the network
        if conditions is None:
            xtc = tensor_utils.concatenate_valid([diffused_x, self._transform_log_snr(log_snr_t)], axis=-1)
        else:
            xtc = tensor_utils.concatenate_valid([diffused_x, self._transform_log_snr(log_snr_t), conditions], axis=-1)
        pred = self.output_projector(self.subnet(xtc, training=training), training=training)

        x_pred = self.convert_prediction_to_x(
            pred=pred, z=diffused_x, alpha_t=alpha_t, sigma_t=sigma_t, log_snr_t=log_snr_t
        )

        # Calculate loss
        if self._loss_type == "noise":
            # convert x to epsilon prediction
            noise_pred = (diffused_x - alpha_t * x_pred) / sigma_t
            loss = weights_for_snr * ops.mean((noise_pred - eps_t) ** 2, axis=-1)
        elif self._loss_type == "velocity":
            # convert x to velocity prediction
            velocity_pred = (alpha_t * diffused_x - x_pred) / sigma_t
            v_t = alpha_t * eps_t - sigma_t * x
            loss = weights_for_snr * ops.mean((velocity_pred - v_t) ** 2, axis=-1)
        elif self._loss_type == "F":
            # convert x to F prediction
            sigma_data = self.noise_schedule.sigma_data if hasattr(self.noise_schedule, "sigma_data") else 1.0
            x1 = ops.sqrt(ops.exp(-log_snr_t) + sigma_data**2) / (ops.exp(-log_snr_t / 2) * sigma_data)
            x2 = (sigma_data * alpha_t) / (ops.exp(-log_snr_t / 2) * ops.sqrt(ops.exp(-log_snr_t) + sigma_data**2))
            f_pred = x1 * x_pred - x2 * diffused_x
            f_t = x1 * x - x2 * diffused_x
            loss = weights_for_snr * ops.mean((f_pred - f_t) ** 2, axis=-1)
        else:
            raise ValueError(f"Unknown loss type: {self._loss_type}")

        # apply sample weight
        loss = weighted_mean(loss, sample_weight)

        base_metrics = super().compute_metrics(x, conditions=conditions, sample_weight=sample_weight, stage=stage)
        return base_metrics | {"loss": loss}
