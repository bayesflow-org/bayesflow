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
    conditional_ot_ratio: float = 0.5,
    partial_ot_factor: float = 1.0,
    **kwargs,
) -> Tensor:
    """
    Log-stabilized version of :py:func:`~bayesflow.utils.optimal_transport.sinkhorn.sinkhorn_plan`.
    About 50% slower than the unstabilized version, so use primarily when you need numerical stability.

    :param x1: Tensor of shape (n, ...)
        Samples from the first distribution.

    :param x2: Tensor of shape (m, ...)
        Samples from the second distribution.

    :param conditions: Optional tensor of shape (m, ...)
        Conditions to be used in conditional optimal transport settings.

    :param regularization: Regularization parameter.
        Controls the standard deviation of the Gaussian kernel.
        Default: 1.0

    :param max_steps: Maximum number of iterations.
        Default: 1000

    :param atol: Absolute tolerance for convergence.
        Default: 1e-5.

    :param conditional_ot_ratio: Ratio which measures the proportion of samples that are considered "potential optimal
        transport candidates". 0.5 is equivalent to no conditioning. [2] recommends a ratio of 0.01.
        Only used if `conditions` is not None.
        Default: 0.01

    :param partial_ot_factor: Proportion of mass to transport in partial optimal transport.
        Default: 1.0 (i.e., balanced OT)

    :return: Tensor of shape (n, m) or (n+1, m+1) if partial=True
        The log transport probabilities.
    """
    if not (0.0 < partial_ot_factor <= 1.0):
        raise ValueError(f"s must be in (0, 1] for partial OT, got {partial_ot_factor}")
    partial = partial_ot_factor < 1.0

    cost = squared_euclidean(x1, x2)

    if regularization <= 0.0:
        raise ValueError(f"regularization must be positive, got {regularization}")

    if conditions is not None and conditional_ot_ratio < 0.5:
        cond_cost = cosine_distance(conditions, conditions)
        cost, w = search_for_conditional_weight(
            M=cost,
            C=cond_cost,
            condition_ratio=conditional_ot_ratio,
            **filter_kwargs(kwargs, search_for_conditional_weight),
        )

    cost_scaled = -cost / regularization
    if partial:
        cost_scaled, a, b = augment_for_partial_ot(
            cost_scaled=cost_scaled,
            regularization=regularization,
            s=partial_ot_factor,
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
