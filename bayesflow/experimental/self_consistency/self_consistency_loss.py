from collections.abc import Callable, Mapping, Sequence
from typing import Literal


import keras

from bayesflow.adapters import Adapter
from bayesflow.types import Tensor
from bayesflow.approximators import Approximator


class SelfConsistencyLoss:
    """The Self-consistency loss as defined by Mishra et al. (2026).

    Implied by the Bayes' theorem, log marginal likelihood can be expressed as:

    log p(y) = log p(theta) + log p(y | theta) - log p(theta | y),

    regardless of the values of theta.
    """

    def __init__(
        self,
        num_samples: int,
        adapter: Adapter | None = None,
        gradient: Sequence[Literal["prior", "posterior", "likelihood"]] = ["posterior"],
        adapted: bool = False,
        parameter_keys: Sequence[str] | None = None,
        data_keys: Sequence[str] | None = None,
        **kwargs,
    ):
        self.prior = None
        self.likelihood = None
        self.posterior = None
        self.num_samples = num_samples
        self.adapter = adapter or Adapter()
        self.gradient = gradient
        self.adapted = adapted
        self.parameter_keys = parameter_keys
        self.data_keys = data_keys

    def __call__(self, data: Mapping[str, Tensor], stage: str) -> Mapping[str, Tensor]:
        samples = self._sample_proposal(data)
        self._determine_keys(samples)

        data = self._reshape_data(data, samples)

        log_ml = self._log_marginal_likelihood(data)
        log_ml = keras.ops.reshape(log_ml, (-1, self.num_samples))

        metrics = {"loss": self._sc_loss(log_ml)}

        return metrics

    def _sample_proposal(self, data: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """Sample parameters from a proposal distribution.
        The log marginal likelihood is evaluated at these parameter samples.
        The SC loss is computed over the log marginal likelihoods over these samples.
        """
        samples = self.posterior.sample(num_samples=self.num_samples, conditions=data, numpy=False)
        samples = keras.tree.map_structure(keras.ops.stop_gradient, samples)
        return samples

    def _reshape_data(self, data: Mapping[str, Tensor], samples: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        # reshape to (batch_size * num_samples,...)
        samples = keras.tree.map_structure(lambda s: keras.ops.reshape(s, (-1, *keras.ops.shape(s)[2:])), samples)
        data = keras.tree.map_structure(lambda s: keras.ops.repeat(s, self.num_samples, axis=0), data)

        data = data | samples

        data = keras.tree.map_structure(lambda s: keras.ops.convert_to_tensor(s, dtype="float32"), data)

        return data

    @staticmethod
    def _sc_loss(log_marginal_likelihood: Tensor) -> Tensor:
        """Variance SC loss"""
        loss = keras.ops.var(log_marginal_likelihood, axis=1)
        loss = keras.ops.mean(loss)

        return loss

    def _log_marginal_likelihood(self, adapted_data: Mapping[str, Tensor]) -> Tensor:
        """Compute log marginal likelihoods via inverse Bayes' theorem:
        log p(y) = log p(theta) + log p (y | theta) - log p (theta | y)
        """
        data, log_det_jac = self.adapter(adapted_data, inverse=True, log_det_jac=True, keras=True)

        log_det_jac_pars, log_det_jac_data = self._collect_log_det_jac_keys(log_det_jac)

        log_prior = self._log_prob_component(self.prior, adapted_data, data, log_det_jac_pars, "prior" in self.gradient)
        log_likelihood = self._log_prob_component(
            self.likelihood, adapted_data, data, log_det_jac_data, "likelihood" in self.gradient
        )
        log_posterior = self._log_prob_component(
            self.posterior, adapted_data, data, log_det_jac_pars, "posterior" in self.gradient
        )

        return log_prior + log_likelihood - log_posterior

    def _determine_keys(self, parameter_samples: Mapping[str, Tensor]) -> None:
        """Determine parameter and data keys (for tracking jacobians)"""
        if not self.parameter_keys:
            self.parameter_keys = list(parameter_samples.keys())

        if not self.data_keys:
            if isinstance(self.likelihood, Approximator) and not self.adapted:
                # TODO: This does not work yet!
                data_samples = self.likelihood.sample(num_samples=1, conditions=parameter_samples, numpy=False)
                self.data_keys = list(data_samples.keys())
            elif self.adapted:
                raise ValueError(
                    ".data_keys must be supplied to compute \
                the self-consistency loss on the adapted space with known likelihood."
                )
            else:
                self.data_keys = []

    def _collect_log_det_jac_keys(self, log_det_jac):
        """Collect jacobian adjustments for parameters and data"""
        log_det_jac_pars = keras.ops.zeros(())
        for key in self.parameter_keys:
            log_det_jac_pars += log_det_jac.get(key, keras.ops.zeros(()))

        log_det_jac_data = keras.ops.zeros(())
        for key in self.data_keys:
            log_det_jac_data += log_det_jac.get(key, keras.ops.zeros(()))

        return log_det_jac_pars, log_det_jac_data

    def _log_prob_component(
        self,
        distribution,
        adapted_data: Mapping[str, Tensor],
        data: Mapping[str, Tensor],
        log_det_jac: Tensor,
        gradient: bool,
    ) -> Tensor:
        """Compute log prob for a distribution component (prior/likelihood/posterior),
        potentially correcting them with Jacobians.
        """

        if isinstance(distribution, Approximator):
            log_prob = distribution.log_prob(adapted_data, numpy=False)
            if not self.adapted:
                log_prob = log_prob - log_det_jac
        elif hasattr(distribution, "log_prob"):
            log_prob = distribution.log_prob(**data)
            if self.adapted:
                log_prob = log_prob + log_det_jac
        elif isinstance(distribution, Callable):
            log_prob = distribution(**data)
            if self.adapted:
                log_prob = log_prob + log_det_jac
        else:
            raise TypeError("distribution must be an Approximator, have a .log_prob method, or be callable.")

        if not gradient:
            log_prob = keras.ops.stop_gradient(log_prob)

        return keras.ops.squeeze(log_prob)

    def attach(self, approximators, adapter):
        self.prior = approximators["prior"]
        self.likelihood = approximators["likelihood"]
        self.posterior = approximators["posterior"]
        self.adapter = adapter
        return self
