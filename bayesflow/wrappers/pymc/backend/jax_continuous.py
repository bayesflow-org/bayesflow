from collections.abc import Callable

import jax.numpy as jnp

from .jax_wrapper import JAXWrapper


class JAXContinuous(JAXWrapper):
    """JAX backend for a :class:`~bayesflow.approximators.ContinuousApproximator`.

    The approximator must be trained in **NLE mode**:
    ``inference_variables`` = observations x,
    ``inference_conditions`` = parameters θ.
    The network models ``log p(x | θ)`` directly.
    """

    def make_log_prob(self) -> Callable:
        flow = self.approximator.inference_network
        std = self.approximator.standardizer

        def log_prob(x_i, *params):
            inf_vars = jnp.atleast_1d(jnp.asarray(x_i))
            inf_cond = jnp.stack([jnp.asarray(p) for p in params], axis=0)

            inf_vars = std.maybe_standardize(inf_vars, key="inference_variables")
            inf_cond = std.maybe_standardize(inf_cond, key="inference_conditions")

            lp = flow.log_prob(inf_vars[None, :], conditions=inf_cond[None, :])
            return jnp.squeeze(lp)

        return log_prob
