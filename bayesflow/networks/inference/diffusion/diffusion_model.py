from collections.abc import Sequence, Mapping
from typing import Any, Literal, Callable

import keras
from keras import ops

from bayesflow.types import Tensor, Shape
from bayesflow.utils import (
    expand_right_as,
    find_network,
    integrate,
    integrate_stochastic,
    jacobian_trace,
    layer_kwargs,
    logging,
    maybe_mask_tensor,
    random_mask,
    randomly_mask_along_axis,
    weighted_mean,
    STOCHASTIC_METHODS,
    DETERMINISTIC_METHODS,
)
from bayesflow.utils.serialization import serialize, serializable

from .schedules.noise_schedule import NoiseSchedule
from .dispatch import find_noise_schedule

from ...inference_network import InferenceNetwork
from ...defaults import TIME_MLP_DEFAULTS, DIFFUSION_INTEGRATE_DEFAULTS


@serializable("bayesflow.networks")
class DiffusionModel(InferenceNetwork):
    """Score-based diffusion model for simulation-based inference (SBI).

    Implements a score-based diffusion model with configurable subnet architecture,
    noise schedule, and prediction/loss types for amortized SBI as described in [1].

    Note that score-based diffusion is the most sluggish of all available samplers,
    so expect slower inference times than flow matching and much slower than
    normalizing flows.

    Parameters
    ----------
    subnet : str, type, or keras.Layer, optional
        A neural network type for the diffusion model, will be instantiated using
        *subnet_kwargs*.  If a string is provided, it should be a registered name
        (e.g., ``"time_mlp"``).  If a type or ``keras.Layer`` is provided, it will
        be directly instantiated with the given *subnet_kwargs*.  Any subnet must
        accept a tuple of tensors ``(target, time, conditions)``.
    noise_schedule : {'edm', 'cosine'} or NoiseSchedule or type, optional
        Noise schedule controlling the diffusion dynamics.  Can be a string
        identifier, a schedule class, or a pre-initialised schedule instance.
        Default is ``"edm"``.
    prediction_type : {'velocity', 'noise', 'F', 'x'}, optional
        Output format of the model's prediction.  Default is ``"F"``.
    loss_type : {'velocity', 'noise', 'F'}, optional
        Loss function used to train the model.  Default is ``"noise"``.
    subnet_kwargs : dict[str, Any], optional
        Additional keyword arguments passed to the subnet constructor.
    schedule_kwargs : dict[str, Any], optional
        Additional keyword arguments passed to the noise schedule constructor.
    integrate_kwargs : dict[str, Any], optional
        Configuration dictionary for the ODE/SDE integrator used at inference time.
    drop_cond_prob : float, optional
        Probability of dropping conditions during training (i.e., classifier-free guidance).
        Default is 0.0.
    drop_target_prob : float, optional
        Probability of dropping target values during training (i.e., learning arbitrary
        distributions). Default is 0.0.
    **kwargs
        Additional keyword arguments passed to the base ``InferenceNetwork``.

    References
    ----------
    [1] Arruda, J., Bracher, N., Köthe, U., Hasenauer, J., & Radev, S. T. (2025).
        Diffusion Models in Simulation-Based Inference: A Tutorial Review.
        arXiv preprint arXiv:2512.20685.
    """

    def __init__(
        self,
        *,
        subnet: str | type | keras.Layer = "time_mlp",
        noise_schedule: Literal["edm", "cosine"] | NoiseSchedule | type = "edm",
        prediction_type: Literal["velocity", "noise", "F", "x"] = "F",
        loss_type: Literal["velocity", "noise", "F"] = "noise",
        subnet_kwargs: dict[str, Any] = None,
        schedule_kwargs: dict[str, Any] = None,
        integrate_kwargs: dict[str, Any] = None,
        drop_cond_prob: float = 0.0,
        drop_target_prob: float = 0.0,
        **kwargs,
    ):
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

        self.integrate_kwargs = DIFFUSION_INTEGRATE_DEFAULTS | (integrate_kwargs or {})
        self.seed_generator = keras.random.SeedGenerator()

        subnet_kwargs = subnet_kwargs or {}
        if subnet == "time_mlp":
            subnet_kwargs = TIME_MLP_DEFAULTS | subnet_kwargs
        self.subnet = find_network(subnet, **subnet_kwargs)

        self.output_projector = None
        self.drop_cond_prob = drop_cond_prob
        self.unconditional_mode = False
        self.drop_target_prob = drop_target_prob

    def compute_metrics(
        self,
        x: Tensor | Sequence[Tensor],
        conditions: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
        **kwargs,
    ) -> dict[str, Tensor]:
        subnet_kwargs = self._collect_mask_kwargs(self._SUBNET_MASK_KEYS, kwargs)

        training = stage == "training"
        noise_schedule_training_stage = stage == "training" or stage == "validation"

        if conditions is not None:
            conditions = randomly_mask_along_axis(conditions, self.drop_cond_prob, seed_generator=self.seed_generator)

        # Sample training diffusion time as a low discrepancy sequence to decrease variance
        u0 = keras.random.uniform(shape=(1,), dtype=ops.dtype(x), seed=self.seed_generator)
        i = ops.arange(0, ops.shape(x)[0], dtype=ops.dtype(x))
        t = (u0 + i / ops.cast(ops.shape(x)[0], dtype=ops.dtype(x))) % 1

        # Calculate the noise level
        log_snr_t = self.noise_schedule.get_log_snr(t, training=noise_schedule_training_stage)
        log_snr_t = expand_right_as(log_snr_t, x)

        alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)
        weights_for_snr = self.noise_schedule.get_weights_for_snr(log_snr_t=log_snr_t)

        # Generate noise vector
        eps_t = keras.random.normal(ops.shape(x), dtype=ops.dtype(x), seed=self.seed_generator)

        # Diffuse x to get noisy input to the network
        diffused_x = alpha_t * x + sigma_t * eps_t

        # Generate optional target dropout mask
        mask_x = random_mask(ops.shape(x), self.drop_target_prob, self.seed_generator)
        diffused_x = maybe_mask_tensor(diffused_x, mask=mask_x, replacement=x)

        # Obtain output of the network and transform to prediction of the clean signal x
        norm_log_snr_t = self._transform_log_snr(log_snr_t)
        subnet_out = self.subnet((diffused_x, norm_log_snr_t, conditions), training=training, **subnet_kwargs)
        pred = self.output_projector(subnet_out)

        x_pred = self.convert_prediction_to_x(
            pred=pred, z=diffused_x, alpha_t=alpha_t, sigma_t=sigma_t, log_snr_t=log_snr_t
        )

        # Finally, compute the loss according to the configured loss type.  Note that the standard weighting
        # functions are defined for the noise prediction loss, so if you use a different loss type, you might want
        # to adjust the weighting accordingly.
        match self._loss_type:
            case "noise":
                noise_pred = (diffused_x - alpha_t * x_pred) / sigma_t
                loss = weights_for_snr * ops.mean(mask_x * (noise_pred - eps_t) ** 2, axis=-1)

            case "velocity":
                velocity_pred = (alpha_t * diffused_x - x_pred) / sigma_t
                v_t = alpha_t * eps_t - sigma_t * x
                loss = weights_for_snr * ops.mean(mask_x * (velocity_pred - v_t) ** 2, axis=-1)

            case "F":
                sigma_data = self.noise_schedule.sigma_data if hasattr(self.noise_schedule, "sigma_data") else 1.0
                x1 = ops.sqrt(ops.exp(-log_snr_t) + sigma_data**2) / (ops.exp(-log_snr_t / 2) * sigma_data)
                x2 = (sigma_data * alpha_t) / (ops.exp(-log_snr_t / 2) * ops.sqrt(ops.exp(-log_snr_t) + sigma_data**2))
                f_pred = x1 * x_pred - x2 * diffused_x
                f_t = x1 * x - x2 * diffused_x
                loss = weights_for_snr * ops.mean(mask_x * (f_pred - f_t) ** 2, axis=-1)

            case _:
                raise ValueError(f"Unknown loss type: {self._loss_type}")

        loss = weighted_mean(loss, sample_weight)

        return {"loss": loss}

    def build(self, xz_shape: Shape, conditions_shape: Shape = None):
        if self.built:
            return

        self.base_distribution.build(xz_shape)

        self.output_projector = keras.layers.Dense(units=xz_shape[-1], bias_initializer="zeros")

        # construct input shape for subnet and subnet projector
        time_shape = (xz_shape[0], 1)  # same batch dims, 1 feature
        self.subnet.build((xz_shape, time_shape, conditions_shape))
        out_shape = self.subnet.compute_output_shape((xz_shape, time_shape, conditions_shape))

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
            "drop_cond_prob": self.drop_cond_prob,
            "drop_target_prob": self.drop_target_prob,
        }
        return base_config | serialize(config)

    def guidance_constraint_term(
        self,
        x: Tensor,
        time: Tensor,
        constraints: Callable | Sequence[Callable],
        guidance_strength: float = 1.0,
        scaling_function: Callable | None = None,
        reduce: Literal["sum", "mean"] = "sum",
    ) -> Tensor:
        """
        Backend-agnostic implementation of:
            `∇_x Σ_k log sigmoid( -s(t) * c_k(x) )`

        Parameters
        ----------
        x : Tensor
            The denoised target at time t.
        time : Tensor
            The time corresponding to x.
        constraints : Callable or Sequence[Callable]
            A single constraint function or a list/tuple of constraint functions.
            Each function should take x as input and return a tensor of constraint values.
        guidance_strength : float, optional
            A positive scaling factor for the guidance term. Default is 1.0.
        scaling_function : Callable, optional
            A function that takes time t as input and returns a scaling factor s(t).
            If None, a default scaling based on the noise schedule is used. Default is None.
        reduce : {'sum', 'mean'}, optional
            Method to reduce the log-probabilities from multiple constraints. Default is 'sum'.

        Returns
        -------
        Tensor
            The computed guidance term of the same shape as zt.
        """

        if not isinstance(constraints, Sequence):
            constraints = [constraints]

        if scaling_function is None:

            def scaling_function(t: Tensor):
                log_snr = self.noise_schedule.get_log_snr(t, training=False)
                alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr)
                return ops.square(alpha_t) / ops.square(sigma_t)

        def objective_fn(z):
            st = scaling_function(time)
            logp = keras.ops.zeros((), dtype=z.dtype)
            for c in constraints:
                ck = c(z)
                logp = logp - keras.ops.softplus(st * ck)
            return keras.ops.sum(logp) if reduce == "sum" else keras.ops.mean(logp)

        backend = keras.backend.backend()

        match backend:
            case "jax":
                import jax

                grad = jax.grad(objective_fn)(x)

            case "tensorflow":
                import tensorflow as tf

                with tf.GradientTape() as tape:
                    tape.watch(x)
                    objective = objective_fn(x)
                grad = tape.gradient(objective, x)

            case "torch":
                import torch

                with torch.enable_grad():
                    x_grad = x.clone().detach().requires_grad_(True)
                    objective = objective_fn(x_grad)
                    grad = torch.autograd.grad(
                        outputs=objective,
                        inputs=x_grad,
                    )[0]

            case _:
                raise NotImplementedError(f"Unsupported backend: {backend}")

        return guidance_strength * grad

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

        match self._prediction_type:
            case "velocity":
                return alpha_t * z - sigma_t * pred

            case "noise":
                return (z - sigma_t * pred) / alpha_t

            case "F":
                sigma_data = getattr(self.noise_schedule, "sigma_data", 1.0)
                x1 = (sigma_data**2 * alpha_t) / (ops.exp(-log_snr_t) + sigma_data**2)
                x2 = ops.exp(-log_snr_t / 2) * sigma_data / ops.sqrt(ops.exp(-log_snr_t) + sigma_data**2)
                return x1 * z + x2 * pred

            case "x":
                return pred

            case "score":
                return (z + sigma_t**2 * pred) / alpha_t

            case _:
                raise ValueError(f"Unknown prediction type {self._prediction_type}.")

    def score(
        self,
        xz: Tensor,
        time: float | Tensor = None,
        log_snr_t: Tensor = None,
        conditions: Tensor = None,
        training: bool = False,
        guidance_constraints: Mapping[str, Any] = None,
        guidance_function: Callable[[Tensor, Tensor], Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Computes the score of the target or latent variable `xz`.

        Parameters
        ----------
        xz : Tensor
            The current state of the latent variable `z`, typically of shape (..., D),
            where D is the dimensionality of the latent space.
        time : float or Tensor
            Scalar or tensor representing the time (or noise level) at which the velocity
            should be computed. Will be broadcasted to xz. If None, log_snr_t must be provided.
        log_snr_t : Tensor
            The log signal-to-noise ratio at time `t`. If None, time must be provided.
        conditions : Tensor, optional
            Conditional inputs to the network, such as conditioning variables
            or encoder outputs. Shape must be broadcastable with `xz`. Default is None.
        training : bool, optional
            Whether the model is in training mode. Affects behavior of dropout, batch norm,
            or other stochastic layers. Default is False.
        guidance_constraints : dict[str, Any], optional
            A dictionary of parameters for computing a guidance constraint term, which is
            added to the score for guided sampling. The specific keys and values depend on
            the implementation of `guidance_constraint_term`.
        guidance_function : Callable[[Tensor, Tensor], Tensor], optional
            A custom function for computing a guidance term, which is added to the score
            for guided sampling. The function should accept the predicted clean signal
            `x_pred` and the current time `time` as inputs and return a tensor of the same
            shape as `xz`.
        **kwargs
            Subnet kwargs (e.g., attention_mask, mask) for the subnet layer.
            Also supports guidance_constraints and guidance_function for custom guidance.

        Returns
        -------
        Tensor
            The velocity tensor of the same shape as `xz`, representing the right-hand
            side of the probability-flow SDE or ODE at the given `time`.
        """
        subnet_kwargs = self._collect_mask_kwargs(self._SUBNET_MASK_KEYS, kwargs)

        if log_snr_t is None:
            log_snr_t = self.noise_schedule.get_log_snr(t=time, training=training)
            log_snr_t = expand_right_as(log_snr_t, xz)
            log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))

        if time is None:
            time = self.noise_schedule.get_t_from_log_snr(log_snr_t, training=training)

        alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)

        norm_log_snr = self._transform_log_snr(log_snr_t)
        subnet_out = self.subnet((xz, norm_log_snr, conditions), training=training, **subnet_kwargs)
        pred = self.output_projector(subnet_out)

        x_pred = self.convert_prediction_to_x(pred=pred, z=xz, alpha_t=alpha_t, sigma_t=sigma_t, log_snr_t=log_snr_t)

        score = (alpha_t * x_pred - xz) / ops.square(sigma_t)

        if guidance_constraints is not None:
            guidance = self.guidance_constraint_term(x=x_pred, time=time, **guidance_constraints)
            score = score + guidance

        if guidance_function is not None:
            guidance = guidance_function(x=x_pred, time=time)
            score = score + guidance

        return score

    def velocity(
        self,
        xz: Tensor,
        time: float | Tensor,
        stochastic_solver: bool,
        conditions: Tensor = None,
        training: bool = False,
        **kwargs,
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

        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))

        score = self.score(xz, log_snr_t=log_snr_t, conditions=conditions, training=training, **kwargs)

        # compute velocity f, g of the SDE or ODE
        f, g_squared = self.noise_schedule.get_drift(log_snr_t=log_snr_t, x=xz, training=training)

        if stochastic_solver:
            # for the SDE: d(z) = [f(z, t) - g(t) ^ 2 * score(z, lambda )] dt + g(t) dW
            out = f - g_squared * score
        else:
            # for the ODE: d(z) = [f(z, t) - 0.5 * g(t) ^ 2 * score(z, lambda )] dt
            out = f - 0.5 * g_squared * score

        # Zero out velocity where target is fixed (during inference only)
        if not training:
            target_mask = kwargs.get("target_mask", None)
            out = maybe_mask_tensor(out, mask=target_mask)

        return out

    def diffusion_term(
        self,
        xz: Tensor,
        time: float | Tensor,
        training: bool = False,
        **kwargs,
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
        g_squared = self.noise_schedule.get_drift(log_snr_t=log_snr_t)
        g = ops.sqrt(g_squared)

        # Zero out diffusion where target is fixed (during inference only)
        if not training:
            target_mask = kwargs.get("target_mask", None)
            g = maybe_mask_tensor(g, mask=target_mask)

        return g

    def _velocity_trace(
        self,
        xz: Tensor,
        time: Tensor,
        conditions: Tensor = None,
        max_steps: int = None,
        training: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        def f(x):
            return self.velocity(
                x, time=time, stochastic_solver=False, conditions=conditions, training=training, **kwargs
            )

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
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        # Note: integrators will cherry-pick necessary kwargs, so
        # we can be general (i.e., sloppy) here
        integrate_kwargs = {"start_time": 0.0, "stop_time": 1.0}
        integrate_kwargs |= self.integrate_kwargs
        integrate_kwargs |= kwargs

        if integrate_kwargs["method"] in STOCHASTIC_METHODS:
            logging.warning(
                "Stochastic methods are not supported for density evaluation."
                " Falling back to tsit5 ODE solver."
                " To suppress this warning, explicitly pass a method from " + str(DETERMINISTIC_METHODS) + "."
            )
            integrate_kwargs["method"] = "tsit5"

        # Apply user-provided target mask if available
        target_mask = kwargs.get("target_mask", None)
        targets_fixed = kwargs.get("targets_fixed", None)
        if target_mask is not None:
            target_mask = keras.ops.broadcast_to(target_mask, keras.ops.shape(x))
            targets_fixed = keras.ops.broadcast_to(targets_fixed, keras.ops.shape(x))
            x = maybe_mask_tensor(x, target_mask, replacement=targets_fixed)

        if self.unconditional_mode and conditions is not None:
            conditions = keras.ops.zeros_like(conditions)
            logging.info("Condition masking is applied: conditions are set to zero.")

        if density:

            def deltas(time, xz):
                v, trace = self._velocity_trace(xz, time=time, conditions=conditions, training=training, **kwargs)
                return {"xz": v, "trace": trace}

            state = {
                "xz": x,
                "trace": ops.zeros(ops.shape(x)[:-1] + (1,), dtype=ops.dtype(x)),
            }
            state = integrate(deltas, state, **integrate_kwargs)

            z = state["xz"]
            log_density = self.base_distribution.log_prob(z) + ops.squeeze(state["trace"], axis=-1)

            return z, log_density

        def deltas(time, xz):
            return {
                "xz": self.velocity(
                    xz, time=time, stochastic_solver=False, conditions=conditions, training=training, **kwargs
                )
            }

        state = {"xz": x}
        state = integrate(deltas, state, **integrate_kwargs)
        z = state["xz"]
        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        # Build integrate kwargs: hardcoded defaults → instance config → call-time overrides
        integrate_kwargs = {"start_time": 1.0, "stop_time": 0.0}
        integrate_kwargs |= self.integrate_kwargs
        integrate_kwargs |= kwargs

        # Apply user-provided target mask if available
        target_mask = kwargs.get("target_mask", None)
        targets_fixed = kwargs.get("targets_fixed", None)
        if target_mask is not None:
            target_mask = keras.ops.broadcast_to(target_mask, keras.ops.shape(z))
            targets_fixed = keras.ops.broadcast_to(targets_fixed, keras.ops.shape(z))
            z = maybe_mask_tensor(z, target_mask, replacement=targets_fixed)

        if self.unconditional_mode and conditions is not None:
            conditions = keras.ops.zeros_like(conditions)
            logging.info("Condition masking is applied: conditions are set to zero.")

        if density:
            if integrate_kwargs["method"] in STOCHASTIC_METHODS:
                logging.warning(
                    "Stochastic methods are not supported for density computation."
                    " Falling back to ODE solver."
                    " Use one of the deterministic methods: " + str(DETERMINISTIC_METHODS) + "."
                )
                integrate_kwargs["method"] = "tsit5"

            def deltas(time, xz):
                v, trace = self._velocity_trace(xz, time=time, conditions=conditions, training=training, **kwargs)
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

        if integrate_kwargs["method"] in STOCHASTIC_METHODS:

            def deltas(time, xz):
                return {
                    "xz": self.velocity(
                        xz, time=time, stochastic_solver=True, conditions=conditions, training=training, **kwargs
                    )
                }

            def diffusion(time, xz):
                return {"xz": self.diffusion_term(xz, time=time, training=training, **kwargs)}

            score_fn = None
            if "corrector_steps" in integrate_kwargs or integrate_kwargs.get("method") == "langevin":

                def score_fn(time, xz):
                    return {"xz": self.score(xz, time=time, conditions=conditions, training=training, **kwargs)}

            state = integrate_stochastic(
                drift_fn=deltas,
                diffusion_fn=diffusion,
                score_fn=score_fn,
                noise_schedule=self.noise_schedule,
                state=state,
                seed=self.seed_generator,
                **integrate_kwargs,
            )
        else:

            def deltas(time, xz):
                return {
                    "xz": self.velocity(
                        xz, time=time, stochastic_solver=False, conditions=conditions, training=training, **kwargs
                    )
                }

            state = integrate(deltas, state, **integrate_kwargs)

        x = state["xz"]
        return x
