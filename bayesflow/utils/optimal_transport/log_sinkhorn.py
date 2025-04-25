import keras

from .. import logging

from .euclidean import euclidean


def log_sinkhorn(x1, x2, seed: int = None, **kwargs):
    """
    Log-stabilized version of :py:func:`~bayesflow.utils.optimal_transport.sinkhorn.sinkhorn`.
    Significantly slower than the unstabilized version, so use only when you need numerical stability.
    """
    log_plan = log_sinkhorn_plan(x1, x2, **kwargs)
    assignments = keras.random.categorical(keras.ops.exp(log_plan), num_samples=1, seed=seed)
    assignments = keras.ops.squeeze(assignments, axis=1)

    return assignments


def log_sinkhorn_plan(x1, x2, regularization: float = 1.0, rtol=1e-5, atol=1e-8, max_steps=None):
    """
    Log-stabilized version of :py:func:`~bayesflow.utils.optimal_transport.sinkhorn.sinkhorn_plan`.
    Significantly slower than the unstabilized version, so use only when you need numerical stability.
    """
    cost = euclidean(x1, x2)

    log_plan = cost / -(regularization * keras.ops.mean(cost) + 1e-16)

    def contains_nans(plan):
        return keras.ops.any(keras.ops.isnan(plan))

    def is_converged(plan):
        # for convergence, the plan should be doubly stochastic
        conv0 = keras.ops.all(keras.ops.isclose(keras.ops.logsumexp(plan, axis=0), 0.0, rtol=rtol, atol=atol))
        conv1 = keras.ops.all(keras.ops.isclose(keras.ops.logsumexp(plan, axis=1), 0.0, rtol=rtol, atol=atol))
        return conv0 & conv1

    def cond(_, plan):
        # break the while loop if the plan contains nans or is converged
        return ~(contains_nans(plan) | is_converged(plan))

    def body(steps, plan):
        # Sinkhorn-Knopp: repeatedly normalize the transport plan along each dimension
        plan = keras.ops.log_softmax(plan, axis=0)
        plan = keras.ops.log_softmax(plan, axis=1)

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
