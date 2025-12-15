import keras

from bayesflow.types import Tensor

from .. import logging

from .euclidean import euclidean
from .partial_ot import augment_for_partial_ot


def log_sinkhorn(x1: Tensor, x2: Tensor, seed: int = None, partial: bool = False, **kwargs) -> Tensor:
    """
    Log-stabilized version of :py:func:`~bayesflow.utils.optimal_transport.sinkhorn.sinkhorn`.
    About 50% slower than the unstabilized version, so use only when you need numerical stability.

    Partial optimal transport can be performed by setting `partial=True` to reduce the effect of misspecified mappings
    in mini-batch settings [1].

    [1] Nguyen et al. (2022) "Improving Mini-batch Optimal Transport via Partial Transportation"
    """
    log_plan = log_sinkhorn_plan(x1, x2, partial=partial, **kwargs)

    if partial:  # Ensure assignments are always among real targets
        log_plan = log_plan[:-1, :-1]

    assignments = keras.random.categorical(log_plan, num_samples=1, seed=seed)
    assignments = keras.ops.squeeze(assignments, axis=1)

    return assignments


def log_sinkhorn_plan(
    x1: Tensor,
    x2: Tensor,
    regularization: float = 1.0,
    rtol=1e-5,
    atol=1e-8,
    max_steps=None,
    partial: bool = False,
    s: float = 0.8,
    dummy_cost: float = 1.0,
) -> Tensor:
    """
    Log-stabilized version of :py:func:`~bayesflow.utils.optimal_transport.sinkhorn.sinkhorn_plan`.
    About 50% slower than the unstabilized version, so use primarily when you need numerical stability.
    """
    cost = euclidean(x1, x2)
    cost_scaled = -cost / regularization

    if partial:
        cost_scaled, a, b = augment_for_partial_ot(
            cost_scaled=cost_scaled, regularization=regularization, s=s, dummy_cost=dummy_cost
        )
        log_a = keras.ops.log(a)
        log_b = keras.ops.log(b)
    else:
        # balanced uniform marginals (scalars)
        n, m = keras.ops.shape(cost_scaled)
        log_a = -keras.ops.log(n)
        log_b = -keras.ops.log(m)

    # initialize transport plan from a gaussian kernel
    log_plan = cost_scaled - keras.ops.max(cost_scaled)

    def contains_nans(plan):
        return keras.ops.any(keras.ops.isnan(plan))

    def is_converged(plan):
        # for convergence, the target marginals must match
        conv0 = keras.ops.all(keras.ops.isclose(keras.ops.logsumexp(plan, axis=0), log_b, rtol=0.0, atol=rtol + atol))
        conv1 = keras.ops.all(keras.ops.isclose(keras.ops.logsumexp(plan, axis=1), log_a, rtol=0.0, atol=rtol + atol))
        return conv0 & conv1

    def cond(_, plan):
        # break the while loop if the plan contains nans or is converged
        return ~(contains_nans(plan) | is_converged(plan))

    def body(steps, plan):
        # Sinkhorn-Knopp: repeatedly normalize the transport plan along each dimension
        plan = plan - keras.ops.logsumexp(plan, axis=0, keepdims=True) + log_b
        plan = plan - keras.ops.logsumexp(plan, axis=1, keepdims=True) + log_a

        return steps + 1, plan

    steps = 0
    steps, log_plan = keras.ops.while_loop(cond, body, (steps, log_plan), maximum_iterations=max_steps)

    def do_nothing():
        pass

    def log_steps():
        msg = "Log-Sinkhorn-Knopp converged after {} steps."

        logging.debug(msg, steps)

    def warn_convergence():
        msg = "Log-Sinkhorn-Knopp did not converge after {} steps."

        logging.warning(msg, max_steps)

    def warn_nans():
        msg = "Log-Sinkhorn-Knopp produced NaNs after {} steps."
        logging.warning(msg, steps)

    keras.ops.cond(contains_nans(log_plan), warn_nans, do_nothing)
    keras.ops.cond(is_converged(log_plan), log_steps, warn_convergence)

    return log_plan
