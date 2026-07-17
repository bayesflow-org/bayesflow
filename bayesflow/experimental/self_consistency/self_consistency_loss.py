from collections.abc import Callable, Mapping, Sequence
from typing import Literal


import keras

from bayesflow.adapters import Adapter
from bayesflow.types import Tensor
from bayesflow.approximators import Approximator
from bayesflow.utils.serialization import deserialize, serialize, serializable


@serializable("bayesflow.experimental")
class SelfConsistencyLoss:
    """Self-consistency loss for joint training of prior, likelihood, and posterior.

    Implements the self-consistency loss from Schmitt et al. (2023) and Mishra et al. (2026). The key identity is
    that for any parameter value theta, Bayes' theorem implies:

        log p(y) = log p(theta) + log p(y | theta) - log p(theta | y)

    If the three components are consistent, the right-hand side should have zero variance
    across samples of theta drawn from any proposal. The SC loss minimizes this variance,
    pushing the components toward mutual consistency.

    By default, theta is sampled from the current posterior (with ``stop_gradient``);
    approximators are therefore trained only through `.log_prob` evaluation rather than
    sampling. Only approximators listed in ``gradient`` receive gradients. By default only the
    posterior is trained through the SC loss.

    Schmitt, M., Ivanova, D. R., Habermann, D., Köthe, U., Bürkner, P. C., & Radev, S. T. (2023).
    Leveraging self-consistency for data-efficient amortized Bayesian inference. arXiv preprint arXiv:2310.04395.

    Mishra, A., Habermann, D., Schmitt, M., Radev, S. T., & Bürkner, P. C. (2025).
    Robust Amortized Bayesian Inference with Self-Consistency Losses on Unlabeled Data. arXiv preprint arXiv:2501.13483.

    Parameters
    ----------
    num_samples : int
        Number of parameter samples drawn per observation to estimate the variance of
        log p(y).
    adapter : Adapter, optional
        Adapter shared across all components, used to map between the data/parameter
        space and the adapted (network) space. If None, an identity adapter is used.
    gradient : sequence of {"prior", "likelihood", "posterior"}, optional
        Which components receive gradients through the SC loss. All others are
        wrapped in ``stop_gradient``. Defaults to ``["posterior"]``.
    adapted : bool, optional
        Determines the space in which the SC loss is computed. Analytic distributions
        (prior, likelihood) are always assumed to be defined in the original data space;
        Approximators are always in the adapted (network) space. This flag controls
        which space all components are converted into before summing:

        - ``False`` (default): compute the loss in original (simulator) space. Approximators have their
          log-probs adjusted by ``-log|det J|`` to convert from adapted to data space;
          analytic distributions need no adjustment.
        - ``True``: compute the loss in adapted space. Analytic distributions have their
          log-probs adjusted by ``+log|det J|`` to convert from data to adapted space;
          Approximators need no adjustment.
    parameter_keys : sequence of str, optional
        Keys in the data dict that correspond to model parameters (theta). If None,
        inferred from the keys of the posterior's sample output on the first call.
    data_keys : sequence of str, optional
        Keys in the data dict that correspond to observed data (y). Required when
        ``adapted=True`` and the likelihood is analytic. If None and the likelihood is
        an Approximator, inferred from its sample output.
        Otherwise defaults to an empty list (no Jacobian correction for data).
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
        """Compute the SC loss for one batch.

        Parameters
        ----------
        data : Mapping[str, Tensor]
            A single batch of adapted data (tensors), containing observation keys.
        stage : str, optional
            Current training stage (e.g., "training", "validation", "inference"). Controls
            the behavior of standardization and some metric computations (default is "training").

        Returns
        -------
        dict with key ``"loss"`` holding the scalar SC loss for this batch.
        """
        samples = self._sample_proposal(data)
        data = self._reshape_data(data, samples)
        self._determine_keys(samples, data)

        log_ml = self._log_marginal_likelihood(data, stage)

        metrics = {"loss": self._sc_loss(log_ml)}

        return metrics

    def _sample_proposal(self, data: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """Sample parameters from a proposal distribution - in this case, the
        posterior approximator.
        The log marginal likelihood is evaluated at these parameter samples.
        The SC loss is computed over the log marginal likelihoods over these samples.
        """
        samples = self.posterior.sample(num_samples=self.num_samples, conditions=data, numpy=False)
        samples = keras.tree.map_structure(keras.ops.stop_gradient, samples)
        return samples

    def _reshape_data(self, data: Mapping[str, Tensor], samples: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
        """Tile observations and flatten parameter samples for batched evaluation.

        ``samples`` from ``_sample_proposal`` have shape ``(batch, num_samples, ...)``.
        After reshaping, both data and samples have batch dimension ``batch * num_samples``,
        so each (observation, parameter) pair can be evaluated independently.
        """
        samples = keras.tree.map_structure(lambda s: keras.ops.reshape(s, (-1, *keras.ops.shape(s)[2:])), samples)
        data = keras.tree.map_structure(lambda s: keras.ops.repeat(s, self.num_samples, axis=0), data)

        data = data | samples

        data = keras.tree.map_structure(lambda s: keras.ops.convert_to_tensor(s, dtype="float32"), data)

        return data

    @staticmethod
    def _sc_loss(log_marginal_likelihood: Tensor) -> Tensor:
        """Variance-based SC loss.

        Parameters
        ----------
        log_marginal_likelihood : Tensor, shape ``(batch, num_samples)``
            Estimates of log p(y) for each (observation, parameter sample) pair.

        Returns
        -------
        Scalar tensor: mean over the batch of the per-observation variance across
        parameter samples. A perfectly consistent model would yield zero variance
        for all observations.
        """
        loss = keras.ops.var(log_marginal_likelihood, axis=1)
        loss = keras.ops.mean(loss)

        return loss

    def _log_marginal_likelihood(self, adapted_data: Mapping[str, Tensor], stage: str) -> Tensor:
        """Estimate log p(y) for each element in the batch via Bayes' theorem:

            log p(y) = log p(theta) + log p(y | theta) - log p(theta | y)

        The adapter is applied in inverse mode to recover the original-space data and
        parameter values together with their log-det-Jacobians, which are used to
        correct the log-prob estimates when components operate in different spaces.

        Returns
        =======
        A 2-D tensor ``(batch_size, num_samples)``: Estimates of log p(y) for each (observation, parameter sample) pair.
        """
        data, log_det_jac = self.adapter(adapted_data, inverse=True, log_det_jac=True)

        log_det_jac_pars, log_det_jac_data = self._collect_log_det_jac_keys(log_det_jac)

        log_prior = self._log_prob_component(
            self.prior, adapted_data, data, log_det_jac_pars, "prior" in self.gradient, stage
        )
        log_likelihood = self._log_prob_component(
            self.likelihood, adapted_data, data, log_det_jac_data, "likelihood" in self.gradient, stage
        )
        log_posterior = self._log_prob_component(
            self.posterior, adapted_data, data, log_det_jac_pars, "posterior" in self.gradient, stage
        )

        log_ml = log_prior + log_likelihood - log_posterior
        log_ml = keras.ops.reshape(log_ml, (-1, self.num_samples))

        return log_ml

    def _determine_keys(self, parameter_samples: Mapping[str, Tensor], data: Mapping[str, Tensor]) -> None:
        """Infer ``parameter_keys`` and ``data_keys`` on the first call if not provided.
        These keys are later used for correcting log prob values with jacobian terms from
        the shared adapter.
        """
        if self.parameter_keys is None:
            self.parameter_keys = list(parameter_samples.keys())

        if self.data_keys is None:
            if isinstance(self.likelihood, Approximator) and self.adapted:
                # Approximate likelihood, loss in adapted space
                # Approximator likelihood already lives in adapted space —> no Jacobian correction needed
                self.data_keys = []
            elif isinstance(self.likelihood, Approximator) and not self.adapted:
                # Approximate likelihood, loss in data space
                # Infer data_keys from a likelihood sample
                data_samples = self.likelihood.sample(num_samples=1, conditions=data, numpy=False)
                self.data_keys = list(data_samples.keys())
            elif self.adapted:
                # Analytic likelihood, loss in adapted space
                # Need Jacobians, but cannot infer data keys
                raise ValueError(
                    "data_keys must be supplied when adapted=True and the likelihood is analytic.",
                    "Set data_keys to the list of observation keys in your data dict.",
                )
            else:
                # Analytic likelihood, loss in data space
                # Likelihood is already in data space -> no Jacobian correction needed
                self.data_keys = []

    def _collect_log_det_jac_keys(self, log_det_jac):
        """Sum Jacobian adjustments separately for parameters and data.

        Returns a pair ``(log_det_jac_pars, log_det_jac_data)`` — each a scalar tensor
        accumulating the log-det-Jacobian contributions for all parameter keys and all
        data keys respectively. Keys absent from ``log_det_jac`` contribute zero.
        """
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
        stage: str,
    ) -> Tensor:
        """Evaluate the log probability for one component (prior, likelihood, or posterior).

        Analytic distributions are always assumed to be defined in data space; Approximators
        are always in adapted space. The ``adapted`` flag on the parent object controls the
        target space — both components are converted into it via Jacobian corrections:

        - **Approximator**: receives ``adapted_data``.
          When ``adapted=False`` (loss in data space): corrected by ``-log_det_jac``.
          When ``adapted=True`` (loss in adapted space): no correction.
        - **Analytic distribution** (has ``.log_prob`` or is callable): receives ``data``.
          When ``adapted=False`` (loss in data space): no correction.
          When ``adapted=True`` (loss in adapted space): corrected by ``+log_det_jac``.

        Parameters
        ----------
        distribution :
            The prior, likelihood, or posterior component.
        adapted_data : Mapping[str, Tensor]
            Data in the adapter's output (network) space.
        data : Mapping[str, Tensor]
            Data in the original (model) space, obtained by inverting the adapter.
        log_det_jac : Tensor
            Accumulated log-det-Jacobian for the relevant keys (parameters or data),
            from the inverse adapter call.
        gradient : bool
            If False, wraps the result in ``stop_gradient`` so this component is not
            trained through the SC loss.
        stage: str

        Returns
        -------
        Tensor containing log_prob on the desired (adapted/unadapted) space.
        """

        if isinstance(distribution, Approximator):
            log_prob = distribution.log_prob(adapted_data, numpy=False, stage=stage)
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

    def get_config(self):
        config = {
            "num_samples": self.num_samples,
            "adapter": self.adapter,
            "gradient": list(self.gradient),
            "adapted": self.adapted,
            "parameter_keys": self.parameter_keys,
            "data_keys": self.data_keys,
        }
        return serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def attach(self, approximators, adapter):
        """Bind the prior, likelihood, posterior approximators and an adapter.

        Called by ``SemiSupervisedApproximator.compile`` so that the loss can access the
        components it needs without them being passed at construction time.

        Parameters
        ----------
        approximators : dict
            Must contain keys ``"prior"``, ``"likelihood"``, and ``"posterior"``.
        adapter : Adapter
            Shared adapter.
        """
        self.prior = approximators["prior"]
        self.likelihood = approximators["likelihood"]
        self.posterior = approximators["posterior"]
        self.adapter = adapter
        return self
