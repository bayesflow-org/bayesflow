from typing import Any, Callable, Literal, Never, Sequence

import numpy as np

from bayesflow.types import LogProb, Shape
from bayesflow.utils.decorators import allow_batch_size


class Workflow:
    def make_simulator(self, simulators_list: Sequence[Callable], meta_fn: Callable) -> Never:
        raise NotImplementedError("Method must be implemented by caller.")

    @allow_batch_size
    def simulate(self, batch_shape: Shape, **kwargs: Any) -> dict[str, np.ndarray]:
        raise NotImplementedError("Method must be implemented by caller.")

    def build_graph(self, *args: Any, **kwargs: Any) -> Never:
        raise NotImplementedError("Method must be implemented by caller.")

    def fit(self, **kwargs: Any) -> Never:
        raise NotImplementedError("Method must be implemented by caller.")

    def sample(self, **kwargs: Any) -> Never:
        raise NotImplementedError("Method must be implemented by caller.")

    def log_prob(self, **kwargs: Any) -> Never:
        raise NotImplementedError("Method must be implemented by caller.")

    def importance_resample(
        self,
        samples,
        log_ratio: LogProb,
        *,
        method: Literal["psis", "is"] = "psis",
        r_eff: float = 1.0,
        rng: int | np.random.Generator | None = None,
        return_diagnostics: bool = False,
    ):
        """Thin workflow wrapper around :func:`bayesflow.utils.importance_resample`.

        For posterior approximation correction, compute the log ratio as

        .. code-block:: python

            log_ratio = lambda samples: (
                prior_log_prob(samples) + likelihood_log_prob(samples) - posterior_log_prob(samples)
            )

        For a prior shift, compute

        .. code-block:: python

            log_ratio = lambda samples: (new_prior_log_prob(samples) - old_prior_log_prob(samples))

        Parameters are forwarded unchanged to the utility function.
        """
        from bayesflow.utils.importance_sampling import importance_resample

        return importance_resample(
            samples,
            log_ratio,
            method=method,
            r_eff=r_eff,
            rng=rng,
            return_diagnostics=return_diagnostics,
        )

    def plot_diagnostics(self, **kwargs: Any) -> Never:
        raise NotImplementedError("Method must be implemented by caller.")

    def compute_diagnostics(self, **kwargs: Any) -> Never:
        raise NotImplementedError("Method must be implemented by caller.")
