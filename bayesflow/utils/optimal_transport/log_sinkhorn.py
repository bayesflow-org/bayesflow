import keras

from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs
from .ot_utils import (
    squared_euclidean,
    cosine_distance,
    augment_for_partial_ot,
    search_for_conditional_weight,
)

from .. import logging


def log_sinkhorn(x1: Tensor, x2: Tensor, conditions: Tensor | None = None, seed: int = None, **kwargs) -> Tensor:
    """
    Log-stabilized version of :py:func:`~bayesflow.utils.optimal_transport.sinkhorn.sinkhorn`.
    About 50% slower than the unstabilized version, so use only when you need numerical stability.

    Partial optimal transport can be performed by setting `partial=True` to reduce the effect of misspecified mappings
    in mini-batch settings [1]. For conditional optimal transport, conditions can be provided along with a
    `condition_ratio` [2].

    [1] Nguyen et al. (2022) "Improving Mini-batch Optimal Transport via Partial Transportation"
    [2] Cheng et al. (2025) "The Curse of Conditions: Analyzing and Improving Optimal Transport for
        Conditional Flow-Based Generation"
    [3] Fluri et al. (2024) "Improving Flow Matching for Simulation-Based Inference"
    """
    log_plan = log_sinkhorn_plan(x1, x2, conditions=conditions, **kwargs)

    assignments = keras.random.categorical(log_plan, num_samples=1, seed=seed)
    assignments = keras.ops.squeeze(assignments, axis=1)

    return assignments


def log_sinkhorn_plan(
    x1: Tensor,
    x2: Tensor,
    conditions: Tensor | None = None,
    regularization: float = 1.0,
    atol: float = 1e-5,
    max_steps: int = 1000,
    condition_ratio: float = 0.5,
    partial_factor: float = 1.0,
    **kwargs,
) -> Tensor:
    """
    Compute a log-stabilized Sinkhorn–Knopp optimal transport plan.

    This function provides a numerically stabilized variant of
    :func:`~bayesflow.utils.optimal_transport.sinkhorn.sinkhorn_plan`. It is
    approximately 50% slower than the unstabilized version and should primarily
    be used when improved numerical stability is required.

    Parameters
    ----------
    x1 : Tensor
        Tensor of shape ``(n, ...)`` containing samples from the first distribution.
    x2 : Tensor
        Tensor of shape ``(m, ...)`` containing samples from the second distribution.
    conditions : Tensor, optional
        Optional tensor of shape ``(m, ...)`` providing conditioning information for
        conditional optimal transport. If ``None``, unconditional optimal transport
        is performed.
    regularization : float, optional
        Entropic regularization parameter controlling the bandwidth (standard
        deviation) of the Gaussian kernel. Default is ``1.0``.
    atol : float, optional
        Absolute tolerance used as the convergence criterion. Default is ``1e-5``.
    max_steps : int, optional
        Maximum number of Sinkhorn iterations. Default is ``1000``.
    condition_ratio : float, optional
        Ratio determining the proportion of samples considered as potential optimal
        transport candidates in conditional optimal transport. A value of ``0.5``
        corresponds to no conditioning; smaller values enforce stronger conditioning.
        Only used if ``conditions`` is not ``None``. Default is ``0.5``.
    partial_factor : float, optional
        Proportion of mass to transport in partial optimal transport. A value of
        ``1.0`` corresponds to balanced optimal transport. Default is ``1.0``.
    **kwargs
        Additional keyword arguments passed to the underlying Sinkhorn routine.

    Returns
    -------
    Tensor
        Tensor of shape ``(n, m)`` containing the log transport probabilities. If
        partial optimal transport is used, the returned tensor may have shape
        ``(n + 1, m + 1)``.
    """

    if not (0.0 < partial_factor <= 1.0):
        raise ValueError(f"s must be in (0, 1] for partial OT, got {partial_factor}")
    partial = partial_factor < 1.0

    cost = squared_euclidean(x1, x2)

    if regularization <= 0.0:
        raise ValueError(f"regularization must be positive, got {regularization}")

    if conditions is not None and condition_ratio < 0.5:
        cond_cost = cosine_distance(conditions, conditions)
        cost, w = search_for_conditional_weight(
            M=cost,
            C=cond_cost,
            condition_ratio=condition_ratio,
            **filter_kwargs(kwargs, search_for_conditional_weight),
        )

    cost_scaled = -cost / regularization
    if partial:
        cost_scaled, a, b = augment_for_partial_ot(
            cost_scaled=cost_scaled,
            regularization=regularization,
            s=partial_factor,
            **filter_kwargs(kwargs, augment_for_partial_ot),
        )
        log_a = keras.ops.log(a)
        log_b = keras.ops.log(b)
        n, m = keras.ops.shape(cost_scaled)
    else:
        # balanced uniform marginals (scalars)
        n, m = keras.ops.shape(cost_scaled)
        log_a = keras.ops.full((n,), -keras.ops.log(keras.ops.cast(n, cost_scaled.dtype)))
        log_b = keras.ops.full((m,), -keras.ops.log(keras.ops.cast(m, cost_scaled.dtype)))

    # log-plan is implicitly: log_plan = cost_scaled + u[:, None] + v[None, :]
    u = keras.ops.zeros((n,), dtype=cost_scaled.dtype)
    v = keras.ops.zeros((m,), dtype=cost_scaled.dtype)

    def contains_nans(_plan):
        return keras.ops.any(keras.ops.isnan(_plan))

    def cond(_, __, ___, _err):
        return _err > atol

    def body(_steps, _u, _v, _err):
        u_next = log_a - keras.ops.logsumexp(cost_scaled + keras.ops.expand_dims(_v, 0), axis=1)
        v_next = log_b - keras.ops.logsumexp(cost_scaled + keras.ops.expand_dims(u_next, 1), axis=0)

        # Error check on dual variable change
        err_next = keras.ops.max(keras.ops.abs(u_next - _u))
        return _steps + 1, u_next, v_next, err_next

    err0 = keras.ops.cast(1e30, cost_scaled.dtype)
    steps, u, v, err = keras.ops.while_loop(cond, body, (0, u, v, err0), maximum_iterations=max_steps)

    # final reconstruction
    log_plan = cost_scaled + keras.ops.expand_dims(u, 1) + keras.ops.expand_dims(v, 0)

    def do_nothing():
        pass

    def log_steps():
        msg = "Log-Sinkhorn-Knopp converged after {} steps."
        logging.debug(msg, steps)

    def warn_convergence():
        msg = "Log-Sinkhorn-Knopp did not converge after {} steps."
        logging.warning(msg, steps)

    def warn_nans():
        msg = "Log-Sinkhorn-Knopp produced NaNs after {} steps."
        logging.warning(msg, steps)

    keras.ops.cond(contains_nans(log_plan), warn_nans, do_nothing)
    keras.ops.cond(cond(None, None, None, err), log_steps, warn_convergence)

    if partial:
        return log_plan[:-1, :-1]

    return log_plan
