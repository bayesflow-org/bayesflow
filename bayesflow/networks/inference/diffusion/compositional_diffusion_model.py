from collections.abc import Mapping
from typing import Literal, Callable, Any

import keras
import numpy as np
from keras import ops

from bayesflow.types import Tensor
from bayesflow.utils import (
    expand_right_as,
    integrate,
    integrate_stochastic,
    logging,
    maybe_mask_tensor,
    DETERMINISTIC_METHODS,
    STOCHASTIC_METHODS,
)
from bayesflow.utils.serialization import serializable

from .diffusion_model import DiffusionModel
from .schedules.noise_schedule import NoiseSchedule


@serializable("bayesflow.networks", disable_module_check=True)
class CompositionalDiffusionModel(DiffusionModel):
    """Compositional Diffusion Model for Amortized Bayesian Inference. Allows to learn a single
    diffusion model one single i.i.d simulations that can perform inference for multiple simulations by leveraging a
    compositional score function as in [1].

    [1] Arruda et al. (2026). Compositional amortized inference for large-scale hierarchical Bayesian models.
     ICLR 2026.
    """

    def __init__(
        self,
        *,
        subnet: str | type | keras.Layer = "time_mlp",
        noise_schedule: Literal["edm", "cosine"] | NoiseSchedule | type = "cosine",
        prediction_type: Literal["velocity", "noise", "F", "x", "score", "potential"] = "velocity",
        loss_type: Literal["velocity", "noise", "F"] = "noise",
        subnet_kwargs: dict[str, any] = None,
        schedule_kwargs: dict[str, any] = None,
        integrate_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        """
        Compositional score-based diffusion model for simulation-based inference (SBI).

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
            Default is ``"cosine"``.
        prediction_type : {'velocity', 'noise', 'F', 'x', 'score', 'potential'}, optional
            Output format of the model's prediction.  Default is ``"velocity"``.
        loss_type : {'velocity', 'noise', 'F'}, optional
            Loss function used to train the model.  Default is ``"noise"``.
        subnet_kwargs : dict[str, Any], optional
            Additional keyword arguments passed to the subnet constructor.
        schedule_kwargs : dict[str, Any], optional
            Additional keyword arguments passed to the noise schedule constructor.
        integrate_kwargs : dict[str, Any], optional
            Configuration dictionary for the ODE/SDE integrator used at inference time.
        drop_target_prob : float, optional
            Probability of dropping target values during training (i.e., learning arbitrary
            distributions). Default is 0.0.
        **kwargs
            Additional keyword arguments passed to the base ``InferenceNetwork``.

        References
        ----------
        [1] Arruda et al. (2026). Compositional amortized inference for large-scale hierarchical Bayesian models.
         ICLR 2026.
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

        self.compositional_bridge_d0 = 1.0
        self.compositional_bridge_d1 = 1.0  # no bridge

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
        compute_prior_score: Callable[[Tensor], Tensor] = None,
        mini_batch_size: int | None = None,
        training: bool = False,
        **kwargs,
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
        compute_prior_score: Callable, optional
            Function to compute the prior score ∇_θ log p(θ). Otherwise, the unconditional score is used.
        mini_batch_size : int or None
            Mini batch size for computing individual scores. If None, use all conditions.
        training : bool, optional
            Whether in training mode
        **kwargs
            Additional keyword arguments passed to the individual score computation.

        Returns
        -------
        Tensor
            Compositional velocity of same shape as input xz
        """
        compositional_score = self.compositional_score(
            xz=xz,
            time=time,
            conditions=conditions,
            compute_prior_score=compute_prior_score,
            mini_batch_size=mini_batch_size,
            training=training,
            **kwargs,
        )

        # Calculate standard noise schedule components
        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))

        # Compute velocity using standard drift-diffusion formulation
        f, g_squared = self.noise_schedule.get_drift(log_snr_t=log_snr_t, x=xz, training=training)

        if stochastic_solver:
            # for the SDE: d(z) = [f(z, t) - g(t) ^ 2 * score(z, lambda )] dt + g(t) dW
            out = f - g_squared * compositional_score
        else:
            # for the ODE: d(z) = [f(z, t) - 0.5 * g(t) ^ 2 * score(z, lambda )] dt
            out = f - 0.5 * g_squared * compositional_score

        # Zero out velocity where target is fixed (during inference only)
        if not training:
            target_mask = kwargs.get("target_mask", None)
            out = maybe_mask_tensor(out, mask=target_mask)

        return out

    def compositional_score(
        self,
        xz: Tensor,
        time: float | Tensor,
        conditions: Tensor,
        compute_prior_score: Callable[[Tensor], Tensor] = None,
        mini_batch_size: int | None = None,
        training: bool = False,
        clip: tuple[float, float] | None = (-3, 3),
        mixture_weight: float = 1.0,
        use_jac: bool = False,
        guidance_constraints: Mapping[str, Any] = None,
        guidance_function: Callable[[Tensor, Tensor], Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """
        Computes the compositional score for multiple datasets.

        Parameters
        ----------
        xz : Tensor
            The current state of the latent variable, shape (n_datasets, n_compositional, ...)
        time : float or Tensor
            Time step for the diffusion process
        conditions : Tensor
            Conditional inputs with compositional structure (n_datasets, n_compositional, ...)
        compute_prior_score: Callable, optional
            Function to compute the prior score ∇_θ log p(θ). Otherwise, the unconditional score is used.
        mini_batch_size : int or None
            Mini batch size for computing individual scores. If None, use all conditions.
        training : bool, optional
            Whether in training mode
        clip: (float, float), optional
            Whether to clip the predicted x for numerical stability at given values.
        mixture_weight : float
            Weighting factor for combining unweighted and weighted scores.
            0 means only weighted, 1 means only unweighted. Only used, if 'use_jac=False'.
        use_jac: bool, optional
            Whether to use the Jacobian-based compositional score instead of the direct sum.
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
            Additional keyword arguments passed to the individual score computation

        Returns
        -------
        Tensor
            Compositional velocity of same shape as input xz
        """
        if conditions is None:
            raise ValueError("Conditions are required for compositional sampling")

        log_snr_t = expand_right_as(self.noise_schedule.get_log_snr(t=time, training=training), xz)
        log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))
        alpha_t, sigma_t = self.noise_schedule.get_alpha_sigma(log_snr_t=log_snr_t)
        time = ops.cast(time, dtype=ops.dtype(xz))

        if not use_jac:
            compositional_score = self._compositional_score_direct(
                xz=xz,
                time=time,
                log_snr_t=log_snr_t,
                alpha_t=alpha_t,
                sigma_t=sigma_t,
                conditions=conditions,
                compute_prior_score=compute_prior_score,
                mini_batch_size=mini_batch_size,
                mixture_weight=mixture_weight,
                training=training,
                **kwargs,
            )
        else:
            # uses a Gaussian approximation, can be very slow
            compositional_score = self._compositional_score_jac(
                xz=xz,
                time=time,
                log_snr_t=log_snr_t,
                sigma_t=sigma_t,
                conditions=conditions,
                compute_prior_score=compute_prior_score,
                mini_batch_size=mini_batch_size,
                training=training,
                **kwargs,
            )

        compositional_score = self.compositional_bridge(time) * compositional_score

        if guidance_constraints is not None or guidance_function is not None:
            # x_pred = (z + sigma_t ** 2 * score) / alpha_t
            x_pred = (xz + sigma_t**2 * compositional_score) / alpha_t

            if guidance_constraints is not None:
                guidance = self.guidance_constraint_term(x=x_pred, time=time, **guidance_constraints)
                compositional_score = compositional_score + guidance

            if guidance_function is not None:
                guidance = guidance_function(x=x_pred, time=time)
                compositional_score = compositional_score + guidance

        compositional_score = self._maybe_clip_score(compositional_score, clip, alpha_t, sigma_t, xz)
        return compositional_score

    def _compositional_score_direct(
        self,
        xz: Tensor,
        time: float | Tensor,
        log_snr_t: Tensor,
        alpha_t: Tensor,
        sigma_t: Tensor,
        conditions: Tensor,
        mixture_weight: float,
        compute_prior_score: Callable[[Tensor], Tensor] = None,
        mini_batch_size: int | None = None,
        eps_var: float = 1e-8,
        training: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        Computes the compositional score for multiple datasets using the formula:
        s_ψ(θ,t,Y) = (1-n)(1-t) ∇_θ log p(θ) + Σᵢ₌₁ⁿ s_ψ(θ,t,yᵢ)
        with possible weighting of the scores.

        Parameters
        ----------
        xz : Tensor
            The current state of the latent variable, shape (n_datasets, n_compositional, ...)
        time : float or Tensor
            Time step for the diffusion process.
        log_snr_t : Tensor
            Log SNR at time t, broadcastable to shape of xz.
        alpha_t : Tensor
            Alpha component of noise schedule at time t, broadcastable to shape of xz.
        sigma_t : Tensor
            Sigma component of noise schedule at time t, broadcastable to shape of xz.
        conditions : Tensor
            Conditional inputs with compositional structure (n_datasets, n_compositional, ...)
        mixture_weight : float
            Weighting factor for combining unweighted and weighted scores.
            0 means only weighted, 1 means only unweighted.
        compute_prior_score: Callable, optional
            Function to compute the prior score ∇_θ log p(θ). Otherwise, the unconditional score is estimated.
        mini_batch_size : int or None
            Mini batch size for computing individual scores. If None, use all conditions.
        eps_var: float
            Small constant added to variance for numerical stability when weighting scores.
        training : bool, optional
            Whether in training mode.
        **kwargs
            Additional keyword arguments passed to the individual score computation.

        Returns
        -------
        Tensor
            Compositional score of same shape as input xz
        """
        # Get shapes for compositional structure
        batch_size, n_compositional = ops.shape(conditions)[:2]

        if mini_batch_size is not None and mini_batch_size < n_compositional:
            # sample random indices for mini-batch processing
            mini_batch_idx = keras.random.shuffle(ops.arange(n_compositional), seed=self.seed_generator)[
                :mini_batch_size
            ]
            conditions_batch = ops.take(conditions, mini_batch_idx, axis=1)
        else:
            conditions_batch = conditions
            mini_batch_size = n_compositional

        needs_network_prior = compute_prior_score is None
        if needs_network_prior:
            zero_cond = ops.zeros_like(ops.take(conditions, 0, axis=1))  # (B, d)
            cond_with_prior = ops.concatenate(
                [conditions_batch, ops.expand_dims(zero_cond, 1)], axis=1
            )  # (B, n_obs+1, d)
            n_total = mini_batch_size + 1
        else:
            cond_with_prior = conditions_batch
            n_total = mini_batch_size
        scale = n_compositional / mini_batch_size

        # expand and flatten compositional dimension for score computation
        dims = tuple(ops.shape(xz)[1:])
        snr_dims = tuple(ops.shape(log_snr_t)[1:])
        conditions_dims = tuple(ops.shape(cond_with_prior)[2:])
        xz_reshaped = ops.reshape(ops.repeat(ops.expand_dims(xz, 1), n_total, axis=1), (batch_size * n_total,) + dims)
        log_snr_reshaped = ops.reshape(
            ops.repeat(ops.expand_dims(log_snr_t, 1), n_total, axis=1),
            (batch_size * n_total,) + snr_dims,
        )
        conditions_flat = ops.reshape(cond_with_prior, (batch_size * n_total,) + conditions_dims)
        scores_flat = self.score(
            xz_reshaped,
            log_snr_t=log_snr_reshaped,
            conditions=conditions_flat,
            training=training,
            **kwargs,
        )
        all_scores = ops.reshape(scores_flat, (batch_size, n_total) + dims)
        individual_scores = all_scores[:, :mini_batch_size]

        if needs_network_prior:
            prior_score = all_scores[:, -1]
        else:
            prior_score = (1.0 - time) * compute_prior_score(xz)

        delta = individual_scores - ops.expand_dims(prior_score, axis=1)
        if mixture_weight < 1:
            # Combined score using compositional formula (1-beta) prior_score + beta posterior_score
            # Per-dimension variance across observations
            var_d = ops.sum((delta - ops.mean(delta, axis=1, keepdims=True)) ** 2, axis=1, keepdims=True) / (
                ops.maximum(mini_batch_size - 1.0, 1.0)
            )
            w_d = 1.0 / (var_d + eps_var)  # (B, 1, m)
            w_d_sum = ops.sum(w_d, axis=1)
            weighted_delta = scale * ops.sum(delta * w_d, axis=1) / w_d_sum
            gamma = ops.square(alpha_t) / ops.square(sigma_t)
            update_delta = gamma * weighted_delta / (1.0 + gamma * scale * w_d_sum)

            update_delta = update_delta * (1 - mixture_weight) + scale * ops.sum(delta, axis=1) * mixture_weight
        else:
            update_delta = scale * ops.sum(delta, axis=1)

        # Combined score using compositional formula: (1-n) prior_score + Σᵢ₌₁ⁿ posterior_score
        compositional_score = prior_score + update_delta
        return compositional_score

    def _compositional_score_jac(
        self,
        xz: Tensor,
        time: float | Tensor,
        log_snr_t: Tensor,
        sigma_t: Tensor,
        conditions: Tensor,
        compute_prior_score: Callable[[Tensor], Tensor] | None = None,
        mini_batch_size: int | None = None,
        training: bool = False,
        regularize_precision: float = 1e-6,
        **kwargs,
    ) -> Tensor:
        """Stabilized version of JAC compositional score (Linhart et al. 2026) with mini-batching from
        (Arruda et al. 2026).

        Precision-weighted compositional posterior score where both individual posterior precisions
        and the prior precision are estimated via the Jacobian of the score network.

        Parameters
        ----------
        xz : Tensor
            The current state of the latent variable, shape (n_datasets, n_compositional, ...)
        time : float or Tensor
            Time step for the diffusion process.
        log_snr_t : Tensor
            Log SNR at time t, broadcastable to shape of xz.
        sigma_t : Tensor
            Sigma component of noise schedule at time t, broadcastable to shape of xz.
        conditions : Tensor
            Conditional inputs with compositional structure (n_datasets, n_compositional, ...)
        compute_prior_score: Callable, optional
            Function to compute the prior score ∇_θ log p(θ). Otherwise, the unconditional score is used.
        mini_batch_size : int or None
            Mini batch size for computing individual scores. If None, use all conditions.
        training : bool, optional
            Whether in training mode.
        regularize_precision : float
            Tikhonov regularization added to Λ before solving for numerical stability.
        **kwargs
            Additional keyword arguments passed to the individual score computation.

        Returns
        -------
        Tensor
            Compositional score of same shape as input xz
        """
        batch_size, n_compositional = ops.shape(conditions)[:2]
        m = ops.shape(xz)[-1]
        upsilon_t = ops.maximum(ops.reshape(sigma_t, (-1,))[0] ** 2, 1e-8)

        if mini_batch_size is not None and mini_batch_size < n_compositional:
            # sample random indices for mini-batch processing
            mini_batch_idx = keras.random.shuffle(ops.arange(n_compositional), seed=self.seed_generator)[
                :mini_batch_size
            ]
            conditions_batch = ops.take(conditions, mini_batch_idx, axis=1)
        else:
            conditions_batch = conditions
            mini_batch_size = n_compositional

        # append prior as extra observation (zero-conditioned)
        zero_cond = ops.zeros_like(ops.take(conditions, 0, axis=1))  # (B, d)
        cond_with_prior = ops.concatenate([conditions_batch, ops.expand_dims(zero_cond, 1)], axis=1)  # (B, n_obs+1, d)

        n_total = mini_batch_size + 1  # observations + prior

        dims = tuple(ops.shape(xz)[1:])
        snr_dims = tuple(ops.shape(log_snr_t)[1:])
        cond_dims = tuple(ops.shape(cond_with_prior)[2:])

        # Repeat xz and log_snr_t for each observation+prior
        xz_rep = ops.reshape(
            ops.repeat(ops.expand_dims(xz, 1), n_total, axis=1),
            (batch_size * n_total,) + dims,
        )
        lsnr_rep = ops.reshape(
            ops.repeat(ops.expand_dims(log_snr_t, 1), n_total, axis=1),
            (batch_size * n_total,) + snr_dims,
        )
        cond_flat = ops.reshape(
            cond_with_prior,
            (batch_size * n_total,) + cond_dims,
        )

        all_scores, all_jacs = self._compute_score_and_jacobian(
            xz=xz_rep,
            log_snr_t=lsnr_rep,
            conditions=cond_flat,
            training=training,
            **kwargs,
        )

        # Reshape: (B, n_total, m) and (B, n_total, m, m)
        all_scores = ops.reshape(all_scores, (batch_size, n_total, m))
        all_jacs = ops.reshape(all_jacs, (batch_size, n_total, m, m))

        # compute P_k = (I + υ J_k)⁻¹ for all k at once
        I_m = ops.eye(m, dtype=ops.dtype(xz))
        I_4d = ops.broadcast_to(
            ops.reshape(I_m, (1, 1, m, m)),
            (batch_size, n_total, m, m),
        )
        A_all = I_4d + upsilon_t * all_jacs
        P_all = ops.inv(A_all)

        # P_k @ s_k for all k: (B, n_total, m, 1) → (B, n_total, m)
        scores_col = ops.expand_dims(all_scores, -1)
        Ps_all = ops.squeeze(ops.matmul(P_all, scores_col), axis=-1)  # (B, n_total, m)

        # split observations and prior
        P_obs = P_all[:, :mini_batch_size]  # (B, n_obs, m, m)
        Ps_obs = Ps_all[:, :mini_batch_size]  # (B, n_obs, m)
        P_lambda = P_all[:, -1]  # (B, m, m)

        # Sum over observations
        sum_P = ops.sum(P_obs, axis=1)  # (B, m, m)
        sum_Ps = ops.sum(Ps_obs, axis=1)  # (B, m)

        # Scale for mini-batch estimator
        if mini_batch_size < n_compositional:
            scale = n_compositional / mini_batch_size
            sum_P = scale * sum_P
            sum_Ps = scale * sum_Ps

        # prior score
        if compute_prior_score is not None:
            s_lambda = (1.0 - time) * compute_prior_score(xz)
            P_lambda_s = ops.squeeze(ops.matmul(P_lambda, ops.expand_dims(s_lambda, -1)), axis=-1)
        else:
            P_lambda_s = Ps_all[:, -1]  # already computed

        # solve: Λ x = s̃
        w_prior = 1 - n_compositional
        I_m_batch = ops.broadcast_to(I_m, (batch_size, m, m))

        lambda_matrix = sum_P + w_prior * P_lambda + regularize_precision * I_m_batch
        s_tilde = sum_Ps + w_prior * P_lambda_s

        compositional_score = self._linsolve_batched(lambda_matrix, s_tilde)
        return compositional_score

    @staticmethod
    def _linsolve_batched(lambda_matrix: Tensor, rhs: Tensor) -> Tensor:
        """Solve Λ x = rhs for x, batched.  Lambda: (B,m,m), rhs: (B,m)."""
        rhs_col = ops.expand_dims(rhs, -1)
        x = ops.solve(lambda_matrix, rhs_col)
        return ops.squeeze(x, axis=-1)

    def _compute_score_and_jacobian(
        self,
        xz: Tensor,
        log_snr_t: Tensor,
        conditions: Tensor,
        training: bool = False,
        **kwargs,
    ) -> tuple[Tensor, Tensor]:
        """Compute score and JAC precision Σ̂⁻¹ for one conditioning input using full Jacobian J = ∇_{θ_t} s(θ_t).

        Returns
        -------
        s : Tensor (B, m)
        prec : Tensor (B, m, m)
            Σ̂⁻¹ = (α_t/υ_t)(I + υ_t J)⁻¹
        """
        m = ops.shape(xz)[-1]
        backend = keras.backend.backend()

        match backend:
            case "jax":
                import jax
                import jax.numpy as jnp

                def phi_fn(x):
                    return self.score(
                        xz=x,
                        log_snr_t=log_snr_t,
                        conditions=conditions,
                        training=training,
                        **kwargs,
                    )

                score, vjp_fn = jax.vjp(phi_fn, xz)

                rows = []
                for i in range(m):
                    e_i = jnp.zeros_like(score).at[:, i].set(1.0)
                    (row_i,) = vjp_fn(e_i)
                    rows.append(jnp.expand_dims(row_i, axis=1))

                jacobian = jnp.concatenate(rows, axis=1)

            case "tensorflow":
                import tensorflow as tf

                with tf.GradientTape(persistent=True) as tape:
                    tape.watch(xz)
                    score = self.score(
                        xz=xz,
                        log_snr_t=log_snr_t,
                        conditions=conditions,
                        training=training,
                        **kwargs,
                    )

                if m <= 32:  # for larger dimension
                    rows = []
                    for i in range(m):
                        e_i = tf.one_hot(tf.fill([tf.shape(score)[0]], i), depth=m, dtype=score.dtype)
                        row_i = tape.gradient(score, xz, output_gradients=e_i)
                        rows.append(tf.expand_dims(row_i, axis=1))
                    jacobian = tf.concat(rows, axis=1)
                else:
                    jacobian = tape.batch_jacobian(score, xz, experimental_use_pfor=False)

                del tape

            case "torch":
                import torch

                with torch.enable_grad():
                    xz_leaf = xz.detach().requires_grad_(True)

                    score = self.score(
                        xz=xz_leaf,
                        log_snr_t=log_snr_t,
                        conditions=conditions,
                        training=training,
                        **kwargs,
                    )

                    rows = []
                    for i in range(m):
                        e_i = torch.zeros_like(score)
                        e_i[:, i] = 1.0
                        (row_i,) = torch.autograd.grad(
                            outputs=score,
                            inputs=xz_leaf,
                            grad_outputs=e_i,
                            create_graph=False,
                            retain_graph=(i < m - 1),
                        )
                        rows.append(row_i.unsqueeze(1))

                    jacobian = torch.cat(rows, dim=1)

                score = score.detach()

            case _:
                raise NotImplementedError(f"JAC Jacobian not implemented for backend '{backend}'. ")
        return score, jacobian

    @staticmethod
    def _maybe_clip_score(compositional_score, clip, alpha_t, sigma_t, xz):
        if clip is not None:
            min_clip, max_clip = clip
            x_pred = (xz + sigma_t**2 * compositional_score) / alpha_t
            x_pred = ops.clip(x_pred, min_clip, max_clip)
            compositional_score = (x_pred * alpha_t - xz) / (sigma_t**2)
        return compositional_score

    def _inverse_compositional(
        self,
        z: Tensor,
        conditions: Tensor,
        compute_prior_score: Callable[[Tensor], Tensor] = None,
        density: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """
        Inverse pass for compositional diffusion sampling.
        """
        integrate_kwargs = {"start_time": 1.0, "stop_time": 0.0}
        integrate_kwargs |= self.integrate_kwargs
        integrate_kwargs |= kwargs

        n_compositional = ops.shape(conditions)[1]
        mini_batch_size = integrate_kwargs.pop("mini_batch_size", int(n_compositional * 0.1))
        if "mini_batch_size" in kwargs:
            kwargs.pop("mini_batch_size")
        if mini_batch_size is None:
            mini_batch_size = n_compositional
        mini_batch_size = max(mini_batch_size, 1)
        if keras.backend.backend() == "jax" and mini_batch_size != n_compositional:
            mini_batch_size = n_compositional
            logging.warning("Setting mini_batch_size to n_compositional as jax does not support mini-batching yet.")

        self.compositional_bridge_d0 = float(
            integrate_kwargs.pop("compositional_bridge_d0", self.compositional_bridge_d0)
        )
        self.compositional_bridge_d1 = float(integrate_kwargs.pop("compositional_bridge_d1", 1.0 / n_compositional))

        # x is sampled from a normal distribution, must be scaled with var 1/n_compositional
        scale_latent = n_compositional * self.compositional_bridge(ops.ones(1))
        z = z / ops.sqrt(ops.cast(scale_latent, dtype=ops.dtype(z)))

        # Apply user-provided target mask if available
        target_mask = kwargs.get("target_mask", None)
        targets_fixed = kwargs.get("targets_fixed", None)
        if target_mask is not None:
            target_mask = keras.ops.broadcast_to(target_mask, keras.ops.shape(z))
            targets_fixed = keras.ops.broadcast_to(targets_fixed, keras.ops.shape(z))
            z = maybe_mask_tensor(z, target_mask, replacement=targets_fixed)

        if density:
            if integrate_kwargs["method"] in STOCHASTIC_METHODS:
                logging.warning(
                    "Stochastic methods are not supported for density computation."
                    " Falling back to ODE solver."
                    " Use one of the deterministic methods: " + str(DETERMINISTIC_METHODS) + "."
                )
                integrate_kwargs["method"] = "tsit5"

            def deltas(time, xz):
                v = self.compositional_velocity(
                    xz,
                    time=time,
                    stochastic_solver=False,
                    conditions=conditions,
                    compute_prior_score=compute_prior_score,
                    mini_batch_size=mini_batch_size,
                    training=training,
                    **kwargs,
                )
                trace = ops.zeros(ops.shape(xz)[:-1] + (1,), dtype=ops.dtype(xz))
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
                    "xz": self.compositional_velocity(
                        xz,
                        time=time,
                        stochastic_solver=True,
                        conditions=conditions,
                        compute_prior_score=compute_prior_score,
                        mini_batch_size=mini_batch_size,
                        training=training,
                        **kwargs,
                    )
                }

            def diffusion(time, xz):
                return {"xz": self.diffusion_term(xz, time=time, training=training, **kwargs)}

            score_fn = None
            if "corrector_steps" in integrate_kwargs or integrate_kwargs.get("method") == "langevin":

                def score_fn(time, xz):
                    return {
                        "xz": self.compositional_score(
                            xz=xz,
                            time=time,
                            conditions=conditions,
                            compute_prior_score=compute_prior_score,
                            mini_batch_size=mini_batch_size,
                            training=training,
                            **kwargs,
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
                        **kwargs,
                    )
                }

            state = integrate(deltas, state, **integrate_kwargs)

        x = state["xz"]
        return x
