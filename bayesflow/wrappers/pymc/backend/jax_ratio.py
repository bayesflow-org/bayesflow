from collections.abc import Callable, Sequence

import jax.numpy as jnp

from jax import jit, vjp, vmap
from pytensor.link.jax.dispatch import jax_funcify

from ..pytensor_ops import RatioLogpOp, RatioLogpVJPOp


@jax_funcify.register(RatioLogpOp)
def ratio_logp_op_jax_funcify(op, **kwargs):
    return op.logp_nojit


class JAXRatio:
    """
    Owns the JAX single-trial log-ratio and the PyTensor Ops wrapping it.

    Parameters
    ----------
    exchangeable:
        If True (default) the likelihood is assumed to factorize over
        i.i.d. observations: the log-ratio is vmapped over trials and
        the Op returns a per-trial vector.
        If False the network consumes the full dataset at once (joint /
        non-exchangeable likelihood): no vmap is applied and the Op
        returns a scalar.
    """

    def __init__(
        self,
        ratio_approximator,
        param_names: Sequence[str],
        *,
        exchangeable: bool = True,
    ):
        self.ratio_approximator = ratio_approximator
        self.param_names = tuple(param_names)
        self.exchangeable = exchangeable

        self.log_ratio = self.make_log_ratio()
        self.logp_nojit, self.logp_jit, self.logp_vjp_jit = self.make_logp_functions()

        self.logp_vjp_op = RatioLogpVJPOp(self.logp_vjp_jit)
        self.logp_op = RatioLogpOp(
            logp_jit=self.logp_jit,
            logp_nojit=self.logp_nojit,
            vjp_op=self.logp_vjp_op,
            scalar_output=not exchangeable,
        )

    def make_log_ratio(self) -> Callable:
        """
        Build the single-trial log-ratio callable.

        Returns
        -------
        Callable
            A function ``(x_i, *params) -> scalar`` that evaluates
            ``log p(x_i | params) / p(x_i)`` for a single observation.
        """
        classifier = self.ratio_approximator.inference_network
        projector = self.ratio_approximator.projector
        std = self.ratio_approximator.standardizer

        def log_ratio(x_i, *params):
            inf_vars = jnp.stack([jnp.asarray(p) for p in params], axis=0)
            inf_conds = jnp.atleast_1d(jnp.asarray(x_i))

            inf_vars = std.maybe_standardize(inf_vars, key="inference_variables")
            inf_conds = std.maybe_standardize(inf_conds, key="inference_conditions")

            inputs = jnp.concatenate([inf_vars, inf_conds], axis=0)
            hidden = classifier(inputs[None, :])
            logits = projector(hidden)

            return jnp.squeeze(logits)

        return log_ratio

    def make_logp_functions(self) -> tuple[Callable, Callable, Callable]:
        """
        Build the JIT-compiled log-ratio and its VJP.

        Returns
        -------
        logp_nojit : Callable
            ``(data, *params) -> array`` — (vmapped) log-ratio without JIT.
        logp_jit : Callable
            JIT-compiled version of ``logp_nojit``.
        logp_vjp_jit : Callable
            JIT-compiled ``(data, *params, gz) -> tuple[array, ...]`` that
            returns the VJP of ``logp_nojit`` with respect to ``params``.

        Notes
        -----
        When ``exchangeable=True`` the log-ratio is vmapped over the trial
        axis so that it accepts a batch of observations and broadcast
        parameter vectors.  When ``exchangeable=False`` the network is
        called once on the full data array and returns a scalar.
        """
        n_params = len(self.param_names)

        if self.exchangeable:
            in_axes = (0,) + (0,) * n_params
            logp_nojit = vmap(self.log_ratio, in_axes=in_axes)
        else:
            logp_nojit = self.log_ratio

        logp_jit = jit(logp_nojit)

        def logp_vjp_nojit(data, *params, gz):
            _, vjp_fn = vjp(logp_nojit, data, *params)
            # Ignore gradient wrt observed data; keep only parameter gradients
            return vjp_fn(gz)[1:]

        logp_vjp_jit = jit(logp_vjp_nojit)

        return logp_nojit, logp_jit, logp_vjp_jit
