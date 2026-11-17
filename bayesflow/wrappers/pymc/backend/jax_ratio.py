from collections.abc import Callable

import jax.numpy as jnp

from .jax_wrapper import JAXWrapper


class JAXRatio(JAXWrapper):
    """JAX backend for a :class:`~bayesflow.estimators.RatioEstimator`.

    Evaluates ``log p(x | θ) / p(x)`` via the classifier + projector heads.
    With ``exchangeable=True`` this is vmapped over observations; with
    ``exchangeable=False`` the full data array is passed at once.
    """

    def make_log_prob(self) -> Callable:
        classifier = self.estimator.inference_network
        projector = self.estimator.projector
        std = self.estimator.standardizer

        def log_prob(x, *params):
            inf_vars = jnp.stack([jnp.asarray(p) for p in params], axis=0)
            inf_conds = jnp.atleast_1d(jnp.asarray(x))

            # No change-of-variables correction needed: log-ratio remains invariant
            inf_vars = std.maybe_standardize(inf_vars, key="inference_variables")
            inf_conds = std.maybe_standardize(inf_conds, key="inference_conditions")

            inputs = jnp.concatenate([inf_vars, inf_conds], axis=0)
            logits = projector(classifier(inputs[None, :]))
            return jnp.squeeze(logits)

        return log_prob
