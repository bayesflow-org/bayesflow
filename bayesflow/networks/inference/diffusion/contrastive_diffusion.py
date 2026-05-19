from collections.abc import Sequence

import keras
from keras import ops

from bayesflow.utils import (
    expand_right_as,
    maybe_mask_tensor,
    expand_tile,
    random_mask,
    randomly_mask_along_axis,
    weighted_mean,
)

from bayesflow.types import Tensor
from .diffusion_model import DiffusionModel


class ContrastiveDiffusion(DiffusionModel):
    def __init__(self, gamma=1.0, K=5, **kwargs):
        super().__init__(**kwargs)
        if gamma <= 0:
            raise ValueError(f"Gamma must be positive, got {gamma}.")
        if gamma == float("inf"):
            raise NotImplementedError("NRE-B is not yet supported.")
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}.")

        self.gamma = gamma
        self.K = K

        self.projector = keras.layers.Dense(units=1)
        self.seed_generator = keras.random.SeedGenerator()

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

        # Generate noise vector
        eps_t = keras.random.normal(ops.shape(x), dtype=ops.dtype(x), seed=self.seed_generator)

        # Diffuse x to get noisy input to the network
        diffused_x = alpha_t * x + sigma_t * eps_t

        # Generate optional target dropout mask
        mask_x = random_mask(ops.shape(x), self.drop_target_prob, self.seed_generator)
        diffused_x = maybe_mask_tensor(diffused_x, mask=mask_x, replacement=x)

        batch_size = keras.ops.shape(diffused_x)[0]

        log_gamma = keras.ops.broadcast_to(keras.ops.log(self.gamma), (batch_size,))
        log_K = keras.ops.broadcast_to(keras.ops.log(self.K), (batch_size,))

        marginal_weight = 1 / (1 + self.gamma)
        joint_weight = self.gamma / (1 + self.gamma)

        # Get (batch_size, K+1, dim) inference variables (theta)
        bootstrap_x = self._sample_from_batch(diffused_x)
        bootstrap_x = keras.ops.concatenate([diffused_x[:, None, :], bootstrap_x], axis=1)

        # Get (batch_size, K, dim) conditions (already resolved from condition builder)
        conditions = expand_tile(conditions, n=self.K, axis=1)
        log_snr_t = expand_tile(log_snr_t, n=self.K, axis=1)

        marginal_logits = self.logits(bootstrap_x[:, 1:, :], conditions, log_snr_t, training=training, **subnet_kwargs)
        joint_logits = self.logits(bootstrap_x[:, :-1, :], conditions, log_snr_t, training=training, **subnet_kwargs)

        # Eq. 7 (https://arxiv.org/abs/2210.06170) - we use a trick for numerical stability:
        # log(K + gamma * sum_{i=1}^{K} exp(h_i)) = log(exp(log K) + sum_{i=1}^{K} exp(h_i + log gamma))
        # so if we absorb log gamma into the network outputs and concatenate log K, we can use logsumexp

        log_numerator_joint = log_gamma + joint_logits[:, 0]
        log_denominator_joint = keras.ops.concatenate([log_gamma[:, None] + joint_logits, log_K[:, None]], axis=-1)
        log_denominator_joint = keras.ops.logsumexp(log_denominator_joint, axis=-1)

        log_numerator_marginal = log_K
        log_denominator_marginal = keras.ops.concatenate(
            [log_gamma[:, None] + marginal_logits, log_K[:, None]], axis=-1
        )
        log_denominator_marginal = keras.ops.logsumexp(log_denominator_marginal, axis=-1)

        joint_loss = log_denominator_joint - log_numerator_joint
        marginal_loss = log_denominator_marginal - log_numerator_marginal

        loss = marginal_weight * marginal_loss + joint_weight * joint_loss
        total_loss = weighted_mean(loss, sample_weight)
        metrics = {"loss": total_loss}
        return metrics

    def _sample_from_batch(self, inference_variables: Tensor) -> Tensor:
        B = keras.ops.shape(inference_variables)[0]

        if isinstance(B, int) and self.K > B - 1:
            num_contrastive = B - 1
        else:
            num_contrastive = self.K

        # Sample from (B, B-1) space — O(B*K) instead of O(B^2)
        scores = keras.random.uniform(
            shape=(B, B - 1),
            dtype="float32",
            seed=self.seed_generator,
        )
        _, idx = keras.ops.top_k(scores, k=num_contrastive)

        # Remap indices >= row index to skip self: [0..i-1] unchanged, [i..B-2] -> [i+1..B-1]
        row = keras.ops.arange(B)[:, None]  # (B, 1)
        idx = idx + keras.ops.cast(idx >= row, dtype=idx.dtype)

        return keras.ops.take(inference_variables, idx, axis=0)

    def logits(self, x: Tensor, conditions: Tensor, norm_log_snr: Tensor, training: bool = False, **kwargs) -> Tensor:
        """Computes logits for K batches of variables-conditions pairs."""
        logits = self.subnet((x, norm_log_snr, conditions), training=training, **kwargs)
        logits = self.projector(logits)
        logits = keras.ops.squeeze(logits, axis=-1)
        return logits

    def score(
        self,
        xz: Tensor,
        time: float | Tensor = None,
        log_snr_t: Tensor = None,
        conditions: Tensor = None,
        training: bool = False,
        guidance_kwargs: any = None,
        **kwargs,
    ) -> Tensor:
        subnet_kwargs = self._collect_mask_kwargs(self._SUBNET_MASK_KEYS, kwargs)

        if log_snr_t is None:
            log_snr_t = self.noise_schedule.get_log_snr(t=time, training=training)
            log_snr_t = expand_right_as(log_snr_t, xz)
            log_snr_t = ops.broadcast_to(log_snr_t, ops.shape(xz)[:-1] + (1,))

        if time is None:
            time = self.noise_schedule.get_t_from_log_snr(log_snr_t, training=training)

        norm_log_snr = self._transform_log_snr(log_snr_t)

        grad_log_ratio = self._compute_grad_of_potential(
            xz,
            norm_log_snr,
            conditions,
            training=training,
            **subnet_kwargs,
        )

        return grad_log_ratio

    def _compositional_score_direct(
        self,
        xz: Tensor,
        time: float | Tensor,
        log_snr_t: Tensor,
        conditions: Tensor,
        seed: keras.random.SeedGenerator = None,
        compute_prior_score=None,
        mini_batch_size: int = None,
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
            The current state of the latent variable, shape (num_datasets, num_items, ...)
        time : float or Tensor
            Time step for the diffusion process.
        log_snr_t : Tensor
            Log SNR at time t, broadcastable to shape of xz.
        conditions : Tensor
            Conditional inputs with compositional structure (num_datasets, num_items, ...)
        seed: keras.random.SeedGenerator or None
            Optional seed for reproducibility.
        compute_prior_score: Callable, optional
            Function to compute the prior score ∇_θ log p(θ). Otherwise, the unconditional score is estimated.
        mini_batch_size : int or None
            Mini batch size for computing individual scores. If None, use all conditions.
        training : bool, optional
            Whether in training mode.
        **kwargs
            Additional keyword arguments passed to the individual score computation.

        Returns
        -------
        Tensor
            Compositional score of same shape as input xz
        """

        batch_size, num_items = ops.shape(conditions)[:2]

        # Sample item indices for mini-batching or keep all items
        if mini_batch_size is not None and mini_batch_size < num_items:
            ranks = keras.random.uniform((batch_size, num_items), seed=seed)
            per_row_idx = ops.top_k(-ranks, mini_batch_size).indices
            conditions_batch = ops.take_along_axis(conditions, per_row_idx[..., None], axis=1)
        else:
            conditions_batch = conditions
            mini_batch_size = num_items

        # Determine scale of summed posterior score
        needs_network_prior = compute_prior_score is None
        if needs_network_prior:
            zero_cond = ops.zeros_like(ops.take(conditions, 0, axis=1))
            cond_with_prior = ops.concatenate([conditions_batch, ops.expand_dims(zero_cond, 1)], axis=1)
            num_total = mini_batch_size + 1
        else:
            cond_with_prior = conditions_batch
            num_total = mini_batch_size
        scale = num_items / mini_batch_size

        # Expand and flatten compositional dimension (i.e., num items) for score computation
        dims = tuple(ops.shape(xz)[1:])
        snr_dims = tuple(ops.shape(log_snr_t)[1:])
        conditions_dims = tuple(ops.shape(cond_with_prior)[2:])
        xz_reshaped = ops.reshape(
            ops.repeat(ops.expand_dims(xz, 1), num_total, axis=1), (batch_size * num_total,) + dims
        )
        log_snr_reshaped = ops.reshape(
            ops.repeat(ops.expand_dims(log_snr_t, 1), num_total, axis=1),
            (batch_size * num_total,) + snr_dims,
        )
        conditions_flat = ops.reshape(cond_with_prior, (batch_size * num_total,) + conditions_dims)
        scores_flat = self.score(
            xz_reshaped,
            log_snr_t=log_snr_reshaped,
            conditions=conditions_flat,
            training=training,
            **kwargs,
        )
        all_scores = ops.reshape(scores_flat, (batch_size, num_total) + dims)
        individual_scores = all_scores[:, :mini_batch_size]

        if needs_network_prior:
            prior_score = all_scores[:, -1]
        else:
            # internally uses a (1-time) weight if prior score has no time argument
            prior_score = compute_prior_score(xz, time)

        # Combined score using compositional formula: (1-n) prior_score + Σᵢ₌₁ⁿ posterior_score
        delta = individual_scores
        update_delta = scale * ops.sum(delta, axis=1)
        compositional_score = prior_score + update_delta

        return compositional_score
