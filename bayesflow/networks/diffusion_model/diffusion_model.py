from collections.abc import Sequence
from typing import Literal

import keras
from keras import ops

from ..inference_network import InferenceNetwork
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
from bayesflow.utils.serialization import serialize, deserialize, serializable

from .schedules.noise_schedule import NoiseSchedule
from .dispatch import find_noise_schedule


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
        "method": "rk45",
        "steps": 100,
    }

    def __init__(
        self,
        *,
        subnet: str | type | keras.Layer = "mlp",
        noise_schedule: Literal["edm", "cosine"] | NoiseSchedule | type = "edm",
        prediction_type: Literal["velocity", "noise", "F", "x"] = "F",
        loss_type: Literal["velocity", "noise", "F"] = "noise",
        subnet_kwargs: dict[str, any] = None,
        schedule_kwargs: dict[str, any] = None,
        integrate_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        """
        Initializes a diffusion model with configurable subnet architecture, noise schedule,
        and prediction/loss types for amortized Bayesian inference.

        Note, that score-based diffusion is the most sluggish of all available samplers,
        so expect slower inference times than flow matching and much slower than normalizing flows.

        Parameters
        ----------
        subnet : str, type or keras.Layer, optional
            Architecture for the transformation network. Can be "mlp", a custom network class, or
            a Layer object, e.g., `bayesflow.networks.MLP(widths=[32, 32])`. Default is "mlp".
        noise_schedule : {'edm', 'cosine'} or NoiseSchedule or type, optional
            Noise schedule controlling the diffusion dynamics. Can be a string identifier,
            a schedule class, or a pre-initialized schedule instance. Default is "edm".
        prediction_type : {'velocity', 'noise', 'F', 'x'}, optional
            Output format of the model's prediction. Default is "F".
        loss_type : {'velocity', 'noise', 'F'}, optional
            Loss function used to train the model. Default is "noise".
        subnet_kwargs : dict[str, any], optional
            Additional keyword arguments passed to the subnet constructor. Default is None.
        schedule_kwargs : dict[str, any], optional
            Additional keyword arguments passed to the noise schedule constructor. Default is None.
        integrate_kwargs : dict[str, any], optional
            Configuration dictionary for integration during training or inference. Default is None.
        concatenate_subnet_input: bool, optional
            Flag for advanced users to control whether all inputs to the subnet should be concatenated
            into a single vector or passed as separate arguments. If set to False, the subnet
            must accept three separate inputs: 'x' (noisy parameters), 't' (log signal-to-noise ratio),
            and optional 'conditions'. Default is True.

        **kwargs
            Additional keyword arguments passed to the base class and internal components.
        """
        super().__init__(base_distribution="normal", **kwargs)

        if prediction_type not in ["noise", "velocity", "F", "x"]:
            raise ValueError(f"Unknown prediction type: {prediction_type}")

        if loss_type not in ["noise", "velocity", "F"]:
            raise ValueError(f"Unknown loss type: {loss_type}")

        if loss_type != "noise":
            logging.warning(
                "The standard schedules have weighting functions defined for the noise prediction loss. "
                "You might want to replace them if you are using a different loss function."
            )

        self._prediction_type = prediction_type
        self._loss_type = loss_type

        schedule_kwargs = schedule_kwargs or {}
        self.noise_schedule = find_noise_schedule(noise_schedule, **schedule_kwargs)
        self.noise_schedule.validate()

        self.integrate_kwargs = self.INTEGRATE_DEFAULT_CONFIG | (integrate_kwargs or {})
        self.seed_generator = keras.random.SeedGenerator()

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "mlp":
            subnet_kwargs = DiffusionModel.MLP_DEFAULT_CONFIG | subnet_kwargs
        self.subnet = find_network(subnet, **subnet_kwargs)
        self._concatenate_subnet_input = kwargs.get("concatenate_subnet_input", True)

        self.output_projector = keras.layers.Dense(units=None, bias_initializer="zeros", name="output_projector")

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        if self.built:
            return

        self.base_distribution.build(xz_shape)

        self.output_projector.units = xz_shape[-1]
        input_shape = list(xz_shape)

        if self._concatenate_subnet_input:
            # construct time vector
            input_shape[-1] += 1
            if conditions_shape is not None:
                input_shape[-1] += conditions_shape[-1]
            input_shape = tuple(input_shape)

            self.subnet.build(input_shape)
            out_shape = self.subnet.compute_output_shape(input_shape)
        else:
            # Multiple separate inputs
            time_shape = tuple(xz_shape[:-1]) + (1,)  # same batch/sequence dims, 1 feature
            self.subnet.build(x_shape=xz_shape, t_shape=time_shape, conditions_shape=conditions_shape)
            out_shape = self.subnet.compute_output_shape(
                x_shape=xz_shape, t_shape=time_shape, conditions_shape=conditions_shape
            )

        self.output_projector.build(out_shape)

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "subnet": self.subnet,
            "noise_schedule": self.noise_schedule,
            "prediction_type": self._prediction_type,
            "loss_type": self._loss_type,
            "integrate_kwargs": self.integrate_kwargs,
            "concatenate_subnet_input": self._concatenate_subnet_input,
            # we do not need to store subnet_kwargs
        }
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def convert_prediction_to_x(
        self, pred: Tensor, z: Tensor, alpha_t: Tensor, sigma_t: Tensor, log_snr_t: Tensor
    ) -> Tensor:
        """
        Converts the neural network prediction into the denoised data `x`, depending on
        the prediction type configured for the model.

        Parameters
        ----------
        pred : Tensor
            The output prediction from the neural network, typically representing noise,
            velocity, or a transformation of the clean signal.
        z : Tensor
            The noisy latent variable `z` to be denoised.
        alpha_t : Tensor
            The noise schedule's scaling factor for the clean signal at time `t`.
        sigma_t : Tensor
            The standard deviation of the noise at time `t`.
        log_snr_t : Tensor
            The log signal-to-noise ratio at time `t`.

        Returns
        -------
        Tensor
            The reconstructed clean signal `x` from the model prediction.
        """
        if self._prediction_type == "velocity":
            return alpha_t * z - sigma_t * pred
        elif self._prediction_type == "noise":
            return (z - sigma_t * pred) / alpha_t
        elif self._prediction_type == "F":
            sigma_data = getattr(self.noise_schedule, "sigma_data", 1.0)
            x1 = (sigma_data**2 * alpha_t) / (ops.exp(-log_snr_t) + sigma_data**2)
            x2 = ops.exp(-log_snr_t / 2) * sigma_data / ops.sqrt(ops.exp(-log_snr_t) + sigma_data**2)
            return x1 * z + x2 * pred
        elif self._prediction_type == "x":
            return pred
        elif self._prediction_type == "score":
            return (z + sigma_t**2 * pred) / alpha_t
        raise ValueError(f"Unknown prediction type {self._prediction_type}.")

    def _apply_subnet(
        self, xz: Tensor, log_snr: Tensor, conditions: Tensor = None, training: bool = False
    ) -> Tensor | tuple[Tensor, Tensor, Tensor]:
        """
        Prepares and passes the input to the subnet either by concatenating the latent variable `xz`,
        the signal-to-noise ratio `log_snr`, and optional conditions or by returning them separately.

        Parameters
        ----------
        xz : Tensor
            The noisy input tensor for the diffusion model, typically of shape (..., D), but can vary.
        log_snr : Tensor
            The log signal-to-noise ratio tensor, typically of shape (..., 1).
        conditions : Tensor, optional
            The optional conditioning tensor (e.g. parameters).
        training : bool, optional
            The training mode flag, which can be used to control behavior during training.

        Returns
        -------
        Tensor
            The output tensor from the subnet.
        """
        if self._concatenate_subnet_input:
            xtc = tensor_utils.concatenate_valid([xz, log_snr, conditions], axis=-1)
            return self.subnet(xtc, training=training)
        else:
            return self.subnet(x=xz, t=log_snr, conditions=conditions, training=training)

    def velocity(
        self,
        xz: Tensor,
        time: float | Tensor,
        stochastic_solver: bool,
        conditions: Tensor = None,
        training: bool = False,
    ) -> Tensor:
        """
        Computes the velocity (i.e., time derivative) of the target or latent variable `xz` for either
        a stochastic differential equation (SDE) or ordinary differential equation (ODE).

        Parameters
        ----------
        xz : Tensor
            The current state of the latent variable `z`, typically of shape (..., D),
            where D is the dimensionality of the latent space.
        time : float or Tensor
            Scalar or tensor representing the time (or noise level) at which the velocity
            should be computed. Will be broadcasted to xz.
        stochastic_solver : bool
            If True, computes the velocity for the stochastic formulation (SDE).
            If False, uses the deterministic formulation (ODE).
        conditions : Tensor, optional
            Conditional inputs to the network, such as conditioning variables
            or encoder outputs. Shape must be broadcastable with `xz`. Default is None.
        training : bool, optional
            Whether the model is in training mode. Affects behavior of dropout, batch norm,
            or other stochastic layers. Default is False.

        Returns
        -------
        Tensor
            The velocity tensor of the same shape as `xz`, representing the right-hand
            side of the SDE or ODE at the given `time`.
        """
        # calculate the current noise level and transform into correct shape
        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))
        alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)

        subnet_out = self._apply_subnet(
            xz, self._transform_log_snr(log_snr_t), conditions=conditions, training=training
        )
        pred = self.output_projector(subnet_out, training=training)

        x_pred = self.convert_prediction_to_x(pred=pred, z=xz, alpha_t=alpha_t, sigma_t=sigma_t, log_snr_t=log_snr_t)

        score = (alpha_t * x_pred - xz) / ops.square(sigma_t)

        # compute velocity f, g of the SDE or ODE
        f, g_squared = self.noise_schedule.get_drift_diffusion(log_snr_t=log_snr_t, x=xz, training=training)

        if stochastic_solver:
            # for the SDE: d(z) = [f(z, t) - g(t) ^ 2 * score(z, lambda )] dt + g(t) dW
            out = f - g_squared * score
        else:
            # for the ODE: d(z) = [f(z, t) - 0.5 * g(t) ^ 2 * score(z, lambda )] dt
            out = f - 0.5 * g_squared * score

        return out

    def diffusion_term(
        self,
        xz: Tensor,
        time: float | Tensor,
        training: bool = False,
    ) -> Tensor:
        """
        Compute the diffusion term (standard deviation of the noise) at a given time.

        Parameters
        ----------
        xz : Tensor
            Input tensor of shape (..., D), typically representing the target or latent variables at given time.
        time : float or Tensor
            The diffusion time step(s). Can be a scalar or a tensor broadcastable to the shape of `xz`.
        training : bool, optional
            Whether to use the training noise schedule (default is False).

        Returns
        -------
        Tensor
            The diffusion term tensor with shape matching `xz` except for the last dimension, which is set to 1.
        """
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
        normalized_snr = (log_snr - log_snr_min) / (log_snr_max - log_snr_min)
        scaled_value = 2 * normalized_snr - 1
        return scaled_value

    def _forward(
        self,
        x: Tensor,
        conditions: Tensor = None,
        density: bool = False,
        training: bool = False,
        compositional: bool = False,
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
        compositional: bool = False,
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
                return {"xz": self.diffusion_term(xz, time=time, training=training)}

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
        i = ops.arange(0, ops.shape(x)[0], dtype=ops.dtype(x))
        t = (u0 + i / ops.cast(ops.shape(x)[0], dtype=ops.dtype(x))) % 1

        # calculate the noise level
        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t, training=noise_schedule_training_stage), x)

        alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)
        weights_for_snr = self.noise_schedule.get_weights_for_snr(log_snr_t=log_snr_t)

        # generate noise vector
        eps_t = keras.random.normal(ops.shape(x), dtype=ops.dtype(x), seed=self.seed_generator)

        # diffuse x
        diffused_x = alpha_t * x + sigma_t * eps_t

        # calculate output of the network
        subnet_out = self._apply_subnet(
            diffused_x, self._transform_log_snr(log_snr_t), conditions=conditions, training=training
        )
        pred = self.output_projector(subnet_out, training=training)

        x_pred = self.convert_prediction_to_x(
            pred=pred, z=diffused_x, alpha_t=alpha_t, sigma_t=sigma_t, log_snr_t=log_snr_t
        )

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

        loss = weighted_mean(loss, sample_weight)

        base_metrics = super().compute_metrics(x, conditions=conditions, sample_weight=sample_weight, stage=stage)
        return base_metrics | {"loss": loss}

    @staticmethod
    def compositional_bridge(time: Tensor) -> Tensor:
        """
        Bridge function for compositional diffusion. In the simplest case, this is just 1.

        Parameters
        ----------
        time: Tensor
            Time step for the diffusion process.

        Returns
        -------
        Tensor
            Bridge function value with same shape as time.

        """
        return ops.ones_like(time)

    def compositional_velocity(
        self,
        xz: Tensor,
        time: float | Tensor,
        stochastic_solver: bool,
        conditions: Tensor,
        mini_batch_idx: Sequence | None,
        training: bool = False,
    ) -> Tensor:
        """
        Computes the compositional velocity for multiple datasets using the formula:
        s_ψ(θ,t,Y) = (1-n)(1-t) ∇_θ log p(θ) + Σᵢ₌₁ⁿ s_ψ(θ,t,yᵢ)

        Parameters
        ----------
        xz : Tensor
            The current state of the latent variable, shape (n_datasets, n_compositional, ...)
        time : float or Tensor
            Time step for the diffusion process
        stochastic_solver : bool
            Whether to use stochastic (SDE) or deterministic (ODE) formulation
        conditions : Tensor
            Conditional inputs with compositional structure (n_datasets, n_compositional, ...)
        mini_batch_idx : Sequence
            Indices for mini-batch selection along the compositional axis.
        training : bool, optional
            Whether in training mode

        Returns
        -------
        Tensor
            Compositional velocity of same shape as input xz
        """
        if conditions is None:
            raise ValueError("Conditions are required for compositional sampling")

        # Get shapes for compositional structure
        n_compositional = ops.shape(conditions)[1]
        n = ops.cast(n_compositional, dtype=ops.dtype(time))
        time_tensor = ops.cast(time, dtype=ops.dtype(xz))

        # Calculate standard noise schedule components
        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))
        alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)

        # Compute individual dataset scores
        if mini_batch_idx is not None:
            conditions_batch = conditions[:, mini_batch_idx]
        else:
            conditions_batch = conditions
        individual_scores = self._compute_individual_scores(xz, log_snr_t, alpha_t, sigma_t, conditions_batch, training)

        # Compute prior score component
        prior_score = self.compute_prior_score(xz)

        # Combine scores using compositional formula, mean over individual scores and scale with n to get sum
        summed_individual_scores = n_compositional * ops.mean(individual_scores, axis=1)

        # Prior contribution
        weighted_prior_score = (1.0 - n) * (1.0 - time_tensor) * prior_score

        # Combined score
        compositional_score = self.compositional_bridge(time_tensor) * (weighted_prior_score + summed_individual_scores)

        # Compute velocity using standard drift-diffusion formulation
        f, g_squared = self.noise_schedule.get_drift_diffusion(log_snr_t=log_snr_t, x=xz, training=training)

        if stochastic_solver:
            # SDE: dz = [f(z,t) - g(t)² * score(z,t)] dt + g(t) dW
            velocity = f - g_squared * compositional_score
        else:
            # ODE: dz = [f(z,t) - 0.5 * g(t)² * score(z,t)] dt
            velocity = f - 0.5 * g_squared * compositional_score

        return velocity

    def _compute_individual_scores(
        self,
        xz: Tensor,
        log_snr_t: Tensor,
        alpha_t: Tensor,
        sigma_t: Tensor,
        conditions: Tensor,
        training: bool,
    ) -> Tensor:
        """
        Compute individual dataset scores s_ψ(θ,t,yᵢ) for each compositional condition.

        Returns
        -------
        Tensor
            Individual scores with shape (n_datasets, n_compositional, ...)
        """
        # Apply subnet to each compositional condition separately
        transformed_log_snr = self._transform_log_snr(log_snr_t)

        # Get shapes
        xz_shape = ops.shape(xz)  # (n_datasets, num_samples, ..., dims)
        conditions_shape = ops.shape(conditions)  # (n_datasets, n_compositional, num_samples, ..., dims)
        n_datasets, n_compositional = conditions_shape[0], conditions_shape[1]
        conditions_dims = tuple(conditions_shape[3:])
        num_samples = xz_shape[1]
        dims = tuple(xz_shape[2:])

        # Expand xz to match compositional structure
        xz_expanded = ops.expand_dims(xz, axis=1)  # (n_datasets, 1, num_samples, ..., dims)
        xz_expanded = ops.broadcast_to(xz_expanded, (n_datasets, n_compositional, num_samples) + dims)

        # Expand noise schedule components to match compositional structure
        log_snr_expanded = ops.expand_dims(transformed_log_snr, axis=1)
        log_snr_expanded = ops.broadcast_to(log_snr_expanded, (n_datasets, n_compositional, num_samples, 1))

        alpha_expanded = ops.expand_dims(alpha_t, axis=1)
        alpha_expanded = ops.broadcast_to(alpha_expanded, (n_datasets, n_compositional, num_samples, 1))

        sigma_expanded = ops.expand_dims(sigma_t, axis=1)
        sigma_expanded = ops.broadcast_to(sigma_expanded, (n_datasets, n_compositional, num_samples, 1))

        # Flatten for subnet application: (n_datasets * n_compositional, num_samples, ..., dims)
        xz_flat = ops.reshape(xz_expanded, (n_datasets * n_compositional, num_samples) + dims)
        log_snr_flat = ops.reshape(log_snr_expanded, (n_datasets * n_compositional, num_samples, 1))
        alpha_flat = ops.reshape(alpha_expanded, (n_datasets * n_compositional, num_samples, 1))
        sigma_flat = ops.reshape(sigma_expanded, (n_datasets * n_compositional, num_samples, 1))
        conditions_flat = ops.reshape(conditions, (n_datasets * n_compositional, num_samples) + conditions_dims)

        # Apply subnet
        subnet_out = self._apply_subnet(xz_flat, log_snr_flat, conditions=conditions_flat, training=training)
        pred = self.output_projector(subnet_out, training=training)

        # Convert prediction to x
        x_pred = self.convert_prediction_to_x(
            pred=pred, z=xz_flat, alpha_t=alpha_flat, sigma_t=sigma_flat, log_snr_t=log_snr_flat
        )

        # Compute score: (α_t * x_pred - z) / σ_t²
        score = (alpha_flat * x_pred - xz_flat) / ops.square(sigma_flat)

        # Reshape back to compositional structure
        score = ops.reshape(score, (n_datasets, n_compositional, num_samples) + dims)
        return score

    def _inverse_compositional(
        self,
        z: Tensor,
        conditions: Tensor = None,
        density: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Inverse pass for compositional diffusion (sampling).
        """
        integrate_kwargs = {"start_time": 1.0, "stop_time": 0.0}
        integrate_kwargs = integrate_kwargs | self.integrate_kwargs
        integrate_kwargs = integrate_kwargs | kwargs
        mini_batch_size = integrate_kwargs.pop("mini_batch_size", None)

        # x is sampled from a normal distribution, must be scaled with var 1/n_compositional
        n_compositional = ops.shape(conditions)[1]
        scale_latent = n_compositional * self.compositional_bridge(ops.ones(1))
        z = z / ops.sqrt(ops.cast(scale_latent, dtype=ops.dtype(z)))

        if mini_batch_size is not None and mini_batch_size < n_compositional:
            # sample random indices for mini-batch processing
            mini_batch_idx = keras.random.shuffle(ops.arange(n_compositional), seed=self.seed_generator)
        else:
            mini_batch_idx = None

        if density:
            if integrate_kwargs["method"] == "euler_maruyama":
                raise ValueError("Stochastic methods are not supported for density computation.")

            def deltas(time, xz):
                v = self.compositional_velocity(
                    xz,
                    time=time,
                    stochastic_solver=False,
                    conditions=conditions,
                    mini_batch_idx=mini_batch_idx,
                    training=training,
                )
                trace = ops.zeros(ops.shape(xz)[:-1] + (1,), dtype=ops.dtype(xz))
                return {"xz": v, "trace": trace}

            state = {
                "xz": z,
                "trace": ops.zeros(ops.shape(z)[:-1] + (1,), dtype=ops.dtype(z)),
            }
            state = integrate(deltas, state, **integrate_kwargs)

            x = state["xz"]
            log_density = self.base_distribution.log_prob(ops.mean(z, axis=1)) - ops.squeeze(state["trace"], axis=-1)
            return x, log_density

        state = {"xz": z}

        if integrate_kwargs["method"] == "euler_maruyama":

            def deltas(time, xz):
                return {
                    "xz": self.compositional_velocity(
                        xz,
                        time=time,
                        stochastic_solver=True,
                        conditions=conditions,
                        mini_batch_idx=mini_batch_idx,
                        training=training,
                    )
                }

            def diffusion(time, xz):
                return {"xz": self.diffusion_term(xz, time=time, training=training)}

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
                    "xz": self.compositional_velocity(
                        xz,
                        time=time,
                        stochastic_solver=False,
                        conditions=conditions,
                        mini_batch_idx=mini_batch_idx,
                        training=training,
                    )
                }

            state = integrate(deltas, state, **integrate_kwargs)

        x = state["xz"]
        return x

    @staticmethod
    def compute_prior_score(xz: Tensor) -> Tensor:
        raise NotImplementedError("Please implement the prior score computation method.")
