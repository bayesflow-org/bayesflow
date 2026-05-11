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

        def log_prob(x, *params):
            inf_vars = jnp.atleast_1d(jnp.asarray(x))
            inf_cond = jnp.stack([jnp.asarray(p) for p in params], axis=0)

            # Apply change-of-variables correction in case of standardization
            inf_vars, log_det_jac = std.maybe_standardize(inf_vars, key="inference_variables", log_det_jac=True)
            inf_cond = std.maybe_standardize(inf_cond, key="inference_conditions")

            lp = flow.log_prob(inf_vars[None, :], conditions=inf_cond[None, :])
            return jnp.squeeze(lp) + log_det_jac

        return log_prob
