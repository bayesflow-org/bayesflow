from typing import Sequence

import numpy as np

from bayesflow.types import Shape
from bayesflow.utils.decorators import allow_batch_size


class Workflow:
    def make_simulator(self, simulators_list: Sequence[callable], meta_fn: callable):
        raise NotImplementedError("Method must be implemented by caller.")

    @allow_batch_size
    def simulate(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        raise NotImplementedError("Method must be implemented by caller.")

    def build_graph(self, *args, **kwargs):
        raise NotImplementedError("Method must be implemented by caller.")

    def fit(self, **kwargs):
        raise NotImplementedError("Method must be implemented by caller.")

    def sample(self, **kwargs):
        raise NotImplementedError("Method must be implemented by caller.")

    def log_prob(self, **kwargs):
        raise NotImplementedError("Method must be implemented by caller.")

    def plot_diagnostics(self, **kwargs):
        raise NotImplementedError("Method must be implemented by caller.")

    def compute_diagnostics(self, **kwargs):
        raise NotImplementedError("Method must be implemented by caller.")
