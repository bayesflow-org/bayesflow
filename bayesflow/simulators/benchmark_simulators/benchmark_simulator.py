import numpy as np

from bayesflow.types import Shape
from bayesflow.utils import batched_call, tree_stack
from bayesflow.utils.decorators import allow_batch_size

from ..simulator import Simulator


class BenchmarkSimulator(Simulator):
    @allow_batch_size
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        """Runs simulated benchmark and returns `batch_size` parameter
        and observation batches

        Parameters
        ----------
        batch_shape: tuple
            Number of parameter-observation batches to simulate.

        Returns
        -------
        dict[str, np.ndarray]: simulated parameters and observables
            with shapes (`batch_size`, ...)
        """

        data = batched_call(self, batch_shape, kwargs=kwargs, flatten=True)
        data = tree_stack(data, axis=0, numpy=True)
        return data

    def prior(self) -> np.ndarray:
        raise NotImplementedError

    def observation_model(self, params: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __call__(self, **kwargs) -> dict[str, np.ndarray]:
        prior_draws = self.prior()
        observables = self.observation_model(prior_draws)
        return dict(parameters=prior_draws.astype(np.float32), observables=observables.astype(np.float32))
