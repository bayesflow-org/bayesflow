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


def sinkhorn(x1: Tensor, x2: Tensor, conditions: Tensor | None = None, seed: int = None, **kwargs) -> Tensor:
    """
    Match elements from ``x2`` onto ``x1`` using the Sinkhorn–Knopp algorithm.

    The Sinkhorn–Knopp algorithm is an iterative procedure that normalizes a cost
    matrix into a transport plan containing assignment probabilities. A permutation
    is then sampled randomly according to this transport plan.

    Partial optimal transport can be enabled to reduce the impact of misspecified
    mappings in mini-batch settings [1]. Conditional optimal transport is supported
    by providing ``conditions`` together with an appropriate conditioning ratio [2].

    Parameters
    ----------
    x1 : Tensor
        Tensor of shape ``(n, ...)`` containing samples from the first distribution.
    x2 : Tensor
        Tensor of shape ``(m, ...)`` containing samples from the second distribution.
    conditions : Tensor, optional
        Optional tensor of shape ``(k, ...)`` providing conditioning information for
        conditional optimal transport. If ``None``, unconditional matching is
        performed. Default is ``None``.
    seed : int or None, optional
        Random seed used for sampling assignment indices. If ``None``, the seed is
        automatically determined in non-compiled contexts. Default is ``None``.
    **kwargs
        Additional keyword arguments forwarded to :func:`sinkhorn_plan`.

    Returns
    -------
    Tensor
        Tensor of shape ``(n,)`` containing assignment indices mapping elements of
        ``x2`` onto ``x1``.

    References
    ----------
    [1] Nguyen et al. (2022). *Improving Mini-batch Optimal Transport via Partial Transportation*.
    [2] Cheng et al. (2025). *The Curse of Conditions: Analyzing and Improving Optimal Transport for
        Conditional Flow-Based Generation*.
    [3] Fluri et al. (2024). *Improving Flow Matching for Simulation-Based Inference*.
    """

    plan = sinkhorn_plan(x1, x2, conditions=conditions, **kwargs)

    # we sample from log(plan) to receive assignments of length n, corresponding to indices of x2
    # such that x2[assignments] matches x1
    assignments = keras.random.categorical(keras.ops.log(plan + 1e-10), num_samples=1, seed=seed)
    assignments = keras.ops.squeeze(assignments, axis=1)

    return assignments


def sinkhorn_plan(
    x1: Tensor,
    x2: Tensor,
    conditions: Tensor | None = None,
    regularization: float = 1.0,
    max_steps: int = 1000,
    atol: float = 1e-5,
    condition_ratio: float = 0.5,
    partial_factor: float = 1.0,
    **kwargs,
) -> Tensor:
    """
    Compute the Sinkhorn–Knopp optimal transport plan.

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
    max_steps : int, optional
        Maximum number of Sinkhorn iterations. Default is ``1000``.
    atol : float, optional
        Absolute tolerance used as the convergence criterion. Default is ``1e-5``.
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
        Tensor of shape ``(n, m)`` containing the optimal transport plan (transport
        probabilities).
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
        a = keras.ops.reshape(a, (-1,))  # (n,)
        b = keras.ops.reshape(b, (-1,))  # (m,)
    else:
        # balanced uniform marginals (scalars)
        n, m = keras.ops.shape(cost_scaled)
        a = keras.ops.ones((n,), dtype=cost_scaled.dtype) / keras.ops.cast(n, cost_scaled.dtype)
        b = keras.ops.ones((m,), dtype=cost_scaled.dtype) / keras.ops.cast(m, cost_scaled.dtype)

    # initialize transport plan from a gaussian kernel
    # (more numerically stable version of keras.ops.exp(-cost/regularization))
    plan = keras.ops.exp(cost_scaled - keras.ops.max(cost_scaled))
    u = keras.ops.ones_like(a)
    v = keras.ops.ones_like(b)
    tiny = keras.ops.cast(1e-12, plan.dtype)

    def contains_nans(_plan):
        return keras.ops.any(keras.ops.isnan(_plan))

    def cond(_, __, ___, _err):
        return _err > atol

    def body(_steps, _u, _v, _err):
        plan_v = keras.ops.matmul(plan, keras.ops.expand_dims(_v, 1))[:, 0] + tiny
        u_new = a / plan_v
        plan_T_u = keras.ops.matmul(keras.ops.transpose(plan), keras.ops.expand_dims(u_new, 1))[:, 0] + tiny
        v_new = b / plan_T_u

        # log-relative change (stable even if u/v span many orders of magnitude)
        du = keras.ops.max(keras.ops.abs(keras.ops.log((u_new + tiny) / (_u + tiny))))
        dv = keras.ops.max(keras.ops.abs(keras.ops.log((v_new + tiny) / (_v + tiny))))
        err_new = keras.ops.maximum(du, dv)

        return _steps + 1, u_new, v_new, err_new

    err0 = keras.ops.cast(1e30, plan.dtype)
    steps, u, v, err = keras.ops.while_loop(cond, body, (0, u, v, err0), maximum_iterations=max_steps)
    plan = (keras.ops.expand_dims(u, 1) * plan) * keras.ops.expand_dims(v, 0)

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
    keras.ops.cond(cond(None, None, None, err), log_steps, warn_convergence)

    if partial:
        plan = plan[:-1, :-1]
    return plan
