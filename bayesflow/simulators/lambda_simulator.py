import numpy as np

from bayesflow.utils import batched_call, filter_kwargs, tree_stack

from .simulator import Simulator
from ..types import Shape


class LambdaSimulator(Simulator):
    """Implements a simulator based on a sampling function."""

    def __init__(self, sample_fn: callable, *, is_batched: bool = False):
        self.sample_fn = sample_fn
        self.is_batched = is_batched

    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        # try to use only valid keyword-arguments
        kwargs = filter_kwargs(kwargs, self.sample_fn)

        if self.is_batched:
            return self.sample_fn(batch_shape, **kwargs)

        data = batched_call(self.sample_fn, batch_shape, kwargs=kwargs, flatten=True)
        data = tree_stack(data, axis=0, numpy=True)

        return data
