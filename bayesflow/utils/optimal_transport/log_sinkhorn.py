from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs
from .ot_utils import (
    euclidean,
    cosine_distance,
    augment_for_partial_ot,
    search_for_conditional_weight,
    auto_regularization,
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
    """
    log_plan = log_sinkhorn_plan(x1, x2, conditions=conditions, **kwargs)

    assignments = keras.random.categorical(log_plan, num_samples=1, seed=seed)
    assignments = keras.ops.squeeze(assignments, axis=1)

    return assignments


def log_sinkhorn_plan(
    x1: Tensor,
    x2: Tensor,
    conditions: Tensor | None = None,
    regularization: Literal["auto"] | float = "auto",
    rtol=1e-5,
    atol=1e-8,
    max_steps: int | None = None,
    condition_ratio: float = 0.5,
    partial: bool = False,
    s: float = 0.8,
    dummy_cost: float = 1.0,
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

    :param max_steps: Maximum number of iterations, or None to run until convergence.
        Default: None

    :param rtol: Relative tolerance for convergence.
        Default: 1e-6.

    :param atol: Absolute tolerance for convergence.
        Default: 1e-8.

    :param condition_ratio: Ratio which measures the proportion of samples that are considered "potential optimal
        transport candidates". 0.5 is equivalent to no conditioning. [2] recommends a ratio of 0.01.
        Only used if `conditions` is not None.
        Default: 0.01

    :param partial: Whether to use partial optimal transport.
        Default: False

    :param s: Proportion of mass to transport in partial optimal transport.
        Must be in (0, 1). Only used if `partial=True`.
        Default: 0.8

    :param dummy_cost: Cost for dummy assignments in partial optimal transport.
        Only used if `partial=True`.
        Default: 1.0

    :return: Tensor of shape (n, m) or (n+1, m+1) if partial=True
        The log transport probabilities.
    """
    if partial and not (0.0 < s < 1.0):
        raise ValueError(f"s must be in (0, 1) for partial OT, got {s}")

    cost = euclidean(x1, x2)

    if regularization == "auto":
        regularization = auto_regularization(cost)
        logging.debug(f"Using regularization {regularization} (auto-tuned) for log Sinkhorn-Knopp OT.")
    elif regularization <= 0.0:
        raise ValueError(f"regularization must be positive, got {regularization}")

    if conditions is not None and condition_ratio < 0.5:
        cond_cost = cosine_distance(conditions, conditions)
        cost, w = search_for_conditional_weight(
            M=cost, C=cond_cost, condition_ratio=condition_ratio, **filter_kwargs(kwargs, search_for_conditional_weight)
        )

    cost_scaled = -cost / regularization

    if partial:
        cost_scaled, a, b = augment_for_partial_ot(
            cost_scaled=cost_scaled, regularization=regularization, s=s, dummy_cost=dummy_cost
        )
        log_a = keras.ops.log(a)
        log_b = keras.ops.log(b)
        log_a_reshape = keras.ops.reshape(log_a, (-1, 1))  # (n+1, 1)
        log_b_reshape = keras.ops.reshape(log_b, (1, -1))  # (1, m+1)
    else:
        # balanced uniform marginals (scalars)
        n, m = keras.ops.shape(cost_scaled)
        log_a_reshape = -keras.ops.log(keras.ops.cast(n, cost_scaled.dtype))  # scalar
        log_b_reshape = -keras.ops.log(keras.ops.cast(m, cost_scaled.dtype))  # scalar

    # initialize transport plan from a gaussian kernel in log space
    log_plan = cost_scaled - keras.ops.max(cost_scaled)

    def contains_nans(plan):
        return keras.ops.any(keras.ops.isnan(plan))

    def is_converged(plan):
        # For convergence, the log marginals must match
        if partial:
            # Compare against vector log marginals
            conv0 = keras.ops.all(
                keras.ops.isclose(keras.ops.logsumexp(plan, axis=0, keepdims=True), log_b_reshape, rtol=rtol, atol=atol)
            )
            conv1 = keras.ops.all(
                keras.ops.isclose(keras.ops.logsumexp(plan, axis=1, keepdims=True), log_a_reshape, rtol=rtol, atol=atol)
            )
        else:
            # Compare against scalar log marginals
            conv0 = keras.ops.all(
                keras.ops.isclose(keras.ops.logsumexp(plan, axis=0), log_b_reshape, rtol=rtol, atol=atol)
            )
            conv1 = keras.ops.all(
                keras.ops.isclose(keras.ops.logsumexp(plan, axis=1), log_a_reshape, rtol=rtol, atol=atol)
            )
        return conv0 & conv1

    def cond(_, plan):
        # break the while loop if the plan contains nans or is converged
        return ~(contains_nans(plan) | is_converged(plan))

    def body(steps, plan):
        # Sinkhorn-Knopp: repeatedly normalize the transport plan along each dimension
        plan = plan - keras.ops.logsumexp(plan, axis=0, keepdims=True) + log_b_reshape
        plan = plan - keras.ops.logsumexp(plan, axis=1, keepdims=True) + log_a_reshape

        return steps + 1, plan

    steps = 0
    steps, log_plan = keras.ops.while_loop(cond, body, (steps, log_plan), maximum_iterations=max_steps)

    steps_value = int(keras.ops.convert_to_numpy(steps))
    has_nans = bool(keras.ops.convert_to_numpy(contains_nans(log_plan)))
    converged = bool(keras.ops.convert_to_numpy(is_converged(log_plan)))

    if has_nans:
        logging.warning(f"Log-Sinkhorn-Knopp produced NaNs after {steps_value} steps.")
    elif converged:
        logging.debug(f"Log-Sinkhorn-Knopp converged after {steps_value} steps.")
    elif max_steps is not None and steps_value >= max_steps:
        logging.warning(f"Log-Sinkhorn-Knopp did not converge after {max_steps} steps. ")
    elif not converged:
        logging.warning("Log-Sinkhorn-Knopp did not converge.")

    if partial:
        return log_plan[:-1, :-1]

    return log_plan
