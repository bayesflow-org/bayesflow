from typing import Literal, Callable

import keras
import numpy as np
from keras import ops

from bayesflow.types import Tensor
from bayesflow.utils import (
    expand_right_as,
    integrate,
    integrate_stochastic,
)
from bayesflow.utils.serialization import serializable
from .diffusion_model import DiffusionModel
from .schedules.noise_schedule import NoiseSchedule


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class CompositionalDiffusionModel(DiffusionModel):
    """Compositional Diffusion Model for Amortized Bayesian Inference. Allows to learn a single
    diffusion model one single i.i.d simulations that can perform inference for multiple simulations by leveraging a
    compositional score function as in [2].

    [1] Score-Based Generative Modeling through Stochastic Differential Equations: Song et al. (2021)
    [2] Compositional Score Modeling for Simulation-Based Inference: Geffner et al. (2023)
    [3] Compositional amortized inference for large-scale hierarchical Bayesian models: Arruda et al. (2025)
    """

    MLP_DEFAULT_CONFIG = {
        "widths": (256, 256, 256, 256, 256),
        "activation": "mish",
        "kernel_initializer": "he_normal",
        "residual": True,
        "dropout": 0.05,
        "spectral_normalization": False,
    }

    INTEGRATE_DEFAULT_CONFIG = {
        "method": "two_step_adaptive",
        "steps": "adaptive",
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
        super().__init__(
            subnet=subnet,
            noise_schedule=noise_schedule,
            prediction_type=prediction_type,
            loss_type=loss_type,
            subnet_kwargs=subnet_kwargs,
            schedule_kwargs=schedule_kwargs,
            integrate_kwargs=integrate_kwargs,
            **kwargs,
        )

    def compositional_bridge(self, time: Tensor) -> Tensor:
        """
        Bridge function for compositional diffusion. In the simplest case, this is just 1 if d0 == d1.
        Otherwise, it can be used to scale the compositional score over time.

        Parameters
        ----------
        time: Tensor
            Time step for the diffusion process.

        Returns
        -------
        Tensor
            Bridge function value with same shape as time.

        """
        return ops.exp(-np.log(self.compositional_bridge_d0 / self.compositional_bridge_d1) * time)

    def compositional_velocity(
        self,
        xz: Tensor,
        time: float | Tensor,
        stochastic_solver: bool,
        conditions: Tensor,
        compute_prior_score: Callable[[Tensor], Tensor],
        mini_batch_size: int | None = None,
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
        compute_prior_score: Callable
            Function to compute the prior score ∇_θ log p(θ).
        mini_batch_size : int or None
            Mini batch size for computing individual scores. If None, use all conditions.
        training : bool, optional
            Whether in training mode

        Returns
        -------
        Tensor
            Compositional velocity of same shape as input xz
        """
        # Calculate standard noise schedule components
        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))

        compositional_score = self.compositional_score(
            xz=xz,
            time=time,
            conditions=conditions,
            compute_prior_score=compute_prior_score,
            mini_batch_size=mini_batch_size,
            training=training,
        )

        # Compute velocity using standard drift-diffusion formulation
        f, g_squared = self.noise_schedule.get_drift_diffusion(log_snr_t=log_snr_t, x=xz, training=training)

        if stochastic_solver:
            # SDE: dz = [f(z,t) - g(t)² * score(z,t)] dt + g(t) dW
            velocity = f - g_squared * compositional_score
        else:
            # ODE: dz = [f(z,t) - 0.5 * g(t)² * score(z,t)] dt
            velocity = f - 0.5 * g_squared * compositional_score

        return velocity

    def compositional_score(
        self,
        xz: Tensor,
        time: float | Tensor,
        conditions: Tensor,
        compute_prior_score: Callable[[Tensor], Tensor],
        mini_batch_size: int | None = None,
        training: bool = False,
    ) -> Tensor:
        """
        Computes the compositional score for multiple datasets using the formula:
        s_ψ(θ,t,Y) = (1-n)(1-t) ∇_θ log p(θ) + Σᵢ₌₁ⁿ s_ψ(θ,t,yᵢ)

        Parameters
        ----------
        xz : Tensor
            The current state of the latent variable, shape (n_datasets, n_compositional, ...)
        time : float or Tensor
            Time step for the diffusion process
        conditions : Tensor
            Conditional inputs with compositional structure (n_datasets, n_compositional, ...)
        compute_prior_score: Callable
            Function to compute the prior score ∇_θ log p(θ).
        mini_batch_size : int or None
            Mini batch size for computing individual scores. If None, use all conditions.
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

        # Calculate standard noise schedule components
        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))

        # Compute individual dataset scores
        if mini_batch_size is not None and mini_batch_size < n_compositional:
            # sample random indices for mini-batch processing
            mini_batch_idx = keras.random.shuffle(ops.arange(n_compositional), seed=self.seed_generator)
            mini_batch_idx = mini_batch_idx[:mini_batch_size]
            conditions_batch = conditions[:, mini_batch_idx]
        else:
            conditions_batch = conditions
        individual_scores = self._compute_individual_scores(xz, log_snr_t, conditions_batch, training)

        # Compute prior score component
        prior_score = compute_prior_score(xz)
        weighted_prior_score = (1.0 - n_compositional) * (1.0 - time) * prior_score

        # Sum individual scores across compositional dimensions
        summed_individual_scores = n_compositional * ops.mean(individual_scores, axis=1)

        # Combined score using compositional formula: (1-n)(1-t)∇log p(θ) + Σᵢ₌₁ⁿ s_ψ(θ,t,yᵢ)
        time_tensor = ops.cast(time, dtype=ops.dtype(xz))
        compositional_score = self.compositional_bridge(time_tensor) * (weighted_prior_score + summed_individual_scores)
        return compositional_score

    def _compute_individual_scores(
        self,
        xz: Tensor,
        log_snr_t: Tensor,
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

        # Expand log_snr_t to match compositional structure
        log_snr_expanded = ops.expand_dims(log_snr_t, axis=1)
        log_snr_expanded = ops.broadcast_to(log_snr_expanded, (n_datasets, n_compositional, num_samples, 1))

        # Flatten for score computation: (n_datasets * n_compositional, num_samples, ..., dims)
        xz_flat = ops.reshape(xz_expanded, (n_datasets * n_compositional, num_samples) + dims)
        log_snr_flat = ops.reshape(log_snr_expanded, (n_datasets * n_compositional, num_samples, 1))
        conditions_flat = ops.reshape(conditions, (n_datasets * n_compositional, num_samples) + conditions_dims)

        # Use standard score function
        scores_flat = self.score(xz_flat, log_snr_t=log_snr_flat, conditions=conditions_flat, training=training)

        # Reshape back to compositional structure
        scores = ops.reshape(scores_flat, (n_datasets, n_compositional, num_samples) + dims)
        return scores

    def _inverse_compositional(
        self,
        z: Tensor,
        conditions: Tensor,
        compute_prior_score: Callable[[Tensor], Tensor],
        density: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Inverse pass for compositional diffusion sampling.
        """
        n_compositional = ops.shape(conditions)[1]
        integrate_kwargs = {"start_time": 1.0, "stop_time": 0.0}
        integrate_kwargs = integrate_kwargs | self.integrate_kwargs
        integrate_kwargs = integrate_kwargs | kwargs
        if keras.backend.backend() == "jax":
            mini_batch_size = integrate_kwargs.pop("mini_batch_size", None)
            if mini_batch_size is not None:
                raise ValueError(
                    "Mini batching is not supported with JAX backend. Set mini_batch_size to None "
                    "or use another backend."
                )
        else:
            mini_batch_size = max(integrate_kwargs.pop("mini_batch_size", int(n_compositional * 0.1)), 1)
        self.compositional_bridge_d0 = float(integrate_kwargs.pop("compositional_bridge_d0", 1.0))
        self.compositional_bridge_d1 = float(integrate_kwargs.pop("compositional_bridge_d1", 1.0))

        # x is sampled from a normal distribution, must be scaled with var 1/n_compositional
        scale_latent = n_compositional * self.compositional_bridge(ops.ones(1))
        z = z / ops.sqrt(ops.cast(scale_latent, dtype=ops.dtype(z)))

        if density:
            if integrate_kwargs["method"] == "euler_maruyama":
                raise ValueError("Stochastic methods are not supported for density computation.")

            def deltas(time, xz):
                v = self.compositional_velocity(
                    xz,
                    time=time,
                    stochastic_solver=False,
                    conditions=conditions,
                    compute_prior_score=compute_prior_score,
                    mini_batch_size=mini_batch_size,
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
                        compute_prior_score=compute_prior_score,
                        mini_batch_size=mini_batch_size,
                        training=training,
                    )
                }

            def diffusion(time, xz):
                return {"xz": self.diffusion_term(xz, time=time, training=training)}

            score_fn = None
            if "corrector_steps" in integrate_kwargs:
                if integrate_kwargs["corrector_steps"] > 0:

                    def score_fn(time, xz):
                        return {
                            "xz": self.compositional_score(
                                xz,
                                time=time,
                                conditions=conditions,
                                compute_prior_score=compute_prior_score,
                                mini_batch_size=mini_batch_size,
                                training=training,
                            )
                        }

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
            integrate_kwargs.pop("corrector_steps", None)

            def deltas(time, xz):
                return {
                    "xz": self.compositional_velocity(
                        xz,
                        time=time,
                        stochastic_solver=False,
                        conditions=conditions,
                        compute_prior_score=compute_prior_score,
                        mini_batch_size=mini_batch_size,
                        training=training,
                    )
                }

            state = integrate(deltas, state, **integrate_kwargs)

        x = state["xz"]
        return x
