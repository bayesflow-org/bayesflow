from typing import Any, Callable, Never, Sequence

import numpy as np

from bayesflow.types import Shape
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

    def plot_diagnostics(self, **kwargs: Any) -> Never:
        raise NotImplementedError("Method must be implemented by caller.")

    def compute_diagnostics(self, **kwargs: Any) -> Never:
        raise NotImplementedError("Method must be implemented by caller.")
