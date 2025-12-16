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


def sinkhorn(x1: Tensor, x2: Tensor, conditions: Tensor | None = None, seed: int = None, **kwargs) -> Tensor:
    """
    Matches elements from x2 onto x1 using the Sinkhorn-Knopp algorithm.

    Sinkhorn-Knopp is an iterative algorithm that repeatedly normalizes the cost matrix into a
    transport plan, containing assignment probabilities.
    The permutation is then sampled randomly according to the transport plan.

    Partial optimal transport can be performed by setting `partial=True` to reduce the effect of misspecified mappings
    in mini-batch settings [1]. For conditional optimal transport, conditions can be provided along with a
    `condition_ratio` [2].

    [1] Nguyen et al. (2022) "Improving Mini-batch Optimal Transport via Partial Transportation"
    [2] Cheng et al. (2025) "The Curse of Conditions: Analyzing and Improving Optimal Transport for
        Conditional Flow-Based Generation"
    [3] Fluri et al. (2024) "Improving Flow Matching for Simulation-Based Inference"

    :param x1: Tensor of shape (n, ...)
        Samples from the first distribution.

    :param x2: Tensor of shape (m, ...)
        Samples from the second distribution.

    :param conditions: Optional tensor of shape (k, ...)
        Conditions to be used in conditional optimal transport settings.
        Default: None

    :param seed: Random seed to use for sampling indices.
        Default: None, which means the seed will be auto-determined for non-compiled contexts.

    :param kwargs:
        Additional keyword arguments that are passed to :py:func:`sinkhorn_plan`.

    :return: Tensor of shape (n,)
        Assignment indices for x2.

    """
    plan = sinkhorn_plan(x1, x2, conditions=conditions, **kwargs)

    # we sample from log(plan) to receive assignments of length n, corresponding to indices of x2
    # such that x2[assignments] matches x1
    assignments = keras.random.categorical(keras.ops.log(plan), num_samples=1, seed=seed)
    assignments = keras.ops.squeeze(assignments, axis=1)

    return assignments


def sinkhorn_plan(
    x1: Tensor,
    x2: Tensor,
    conditions: Tensor | None = None,
    regularization: Literal["auto"] | float = "auto",
    max_steps: int | None = None,
    rtol: float = 1e-5,
    atol: float = 1e-8,
    condition_ratio: float = 0.5,
    partial_s: float = 1.0,
    dummy_cost: float = 1.0,
    **kwargs,
) -> Tensor:
    """
    Computes the Sinkhorn-Knopp optimal transport plan.

    :param x1: Tensor of shape (n, ...)
        Samples from the first distribution.

    :param x2: Tensor of shape (m, ...)
        Samples from the second distribution.

    :param conditions: Optional tensor of shape (m, ...)
        Conditions to be used in conditional optimal transport settings.

    :param regularization: Regularization parameter.
        Controls the standard deviation of the Gaussian kernel.

    :param max_steps: Maximum number of iterations, or None to run until convergence.
        Default: None

    :param rtol: Relative tolerance for convergence.
        Default: 1e-5.

    :param atol: Absolute tolerance for convergence.
        Default: 1e-8.

    :param condition_ratio: Ratio which measures the proportion of samples that are considered “potential optimal
        transport candidates”. 0.5 is equivalent to no conditioning. [2] recommends a ratio of 0.01.
        Only used if `conditions` is not None.
        Default: 0.0

    :param partial_s: Proportion of mass to transport in partial optimal transport.
        Default: 1.0 (i.e., balanced OT)

    :param dummy_cost: Cost for dummy assignments in partial optimal transport.
        Only used if `partial=True`.
        Default: 1.0

    :return: Tensor of shape (n, m)
        The transport probabilities.
    """
    partial = False
    if not (0.0 < partial_s <= 1.0):
        raise ValueError(f"s must be in (0, 1] for partial OT, got {partial_s}")
    elif partial_s < 1.0:
        partial = True

    cost = euclidean(x1, x2)

    if regularization == "auto":
        regularization = auto_regularization(cost)
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
            cost_scaled=cost_scaled, regularization=regularization, s=partial_s, dummy_cost=dummy_cost
        )
        # a and b are vectors of shape (n,) and (m,)
        a_reshape = keras.ops.reshape(a, (-1, 1))  # (n, 1)
        b_reshape = keras.ops.reshape(b, (1, -1))  # (1, m)
    else:
        # balanced uniform marginals (scalars)
        n, m = keras.ops.shape(cost_scaled)
        a_reshape = 1.0 / keras.ops.cast(n, cost_scaled.dtype)
        b_reshape = 1.0 / keras.ops.cast(m, cost_scaled.dtype)

    # initialize transport plan from a gaussian kernel
    # (more numerically stable version of keras.ops.exp(-cost/regularization))
    plan = keras.ops.exp(cost_scaled - keras.ops.max(cost_scaled))

    def contains_nans(plan):
        return keras.ops.any(keras.ops.isnan(plan))

    def is_converged(plan):
        # for convergence, the target marginals must match
        if partial:
            # Check against vector marginals
            conv0 = keras.ops.all(
                keras.ops.isclose(keras.ops.sum(plan, axis=0, keepdims=True), b_reshape, rtol=rtol, atol=atol)
            )
            conv1 = keras.ops.all(
                keras.ops.isclose(keras.ops.sum(plan, axis=1, keepdims=True), a_reshape, rtol=rtol, atol=atol)
            )
        else:
            # Check against scalar marginals
            conv0 = keras.ops.all(keras.ops.isclose(keras.ops.sum(plan, axis=0), b_reshape, rtol=rtol, atol=atol))
            conv1 = keras.ops.all(keras.ops.isclose(keras.ops.sum(plan, axis=1), a_reshape, rtol=rtol, atol=atol))
        return conv0 & conv1

    def cond(_, plan):
        # break the while loop if the plan contains nans or is converged
        return ~(contains_nans(plan) | is_converged(plan))

    def body(steps, plan):
        # Sinkhorn-Knopp: repeatedly normalize the transport plan along each dimension
        plan = plan / keras.ops.sum(plan, axis=0, keepdims=True) * b_reshape
        plan = plan / keras.ops.sum(plan, axis=1, keepdims=True) * a_reshape

        return steps + 1, plan

    steps = 0
    steps, plan = keras.ops.while_loop(cond, body, (steps, plan), maximum_iterations=max_steps)

    def do_nothing():
        pass

    def log_steps():
        msg = "Sinkhorn-Knopp converged after {} steps."
        logging.debug(msg, steps)

    def warn_convergence():
        msg = "Sinkhorn-Knopp did not converge after {} steps."
        logging.warning(msg, steps)

    def warn_nans():
        msg = "Sinkhorn-Knopp produced NaNs after {} steps."
        logging.warning(msg, steps)

    keras.ops.cond(contains_nans(plan), warn_nans, do_nothing)
    keras.ops.cond(is_converged(plan), log_steps, warn_convergence)

    if partial:
        plan = plan[:-1, :-1]
    return plan
