from collections.abc import Callable, Sequence

import numpy as np

from bayesflow.utils import batched_call, filter_kwargs, tree_stack
from bayesflow.utils.decorators import allow_batch_size

from .simulator import Simulator
from ..types import Shape


class LambdaSimulator(Simulator):
    """Implements a simulator based on a sampling function."""

    def __init__(self, sample_fn: Callable[[Sequence[int]], dict[str, any]], *, is_batched: bool = False):
        """
        Initialize a simulator based on a simple callable function

        Parameters
        ----------
        sample_fn : Callable[[Sequence[int]], dict[str, any]]
            A function that generates samples. It should accept `batch_shape` as its first argument
            (if `is_batched=True`), followed by keyword arguments.
        is_batched : bool, optional
            Whether the `sample_fn` is implemented to handle batched sampling directly.
            If False, `sample_fn` will be called once per sample and results will be stacked.
            Default is False.
        """
        self.sample_fn = sample_fn
        self.is_batched = is_batched

    @allow_batch_size
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        """
        Sample using the wrapped sampling function.

        Parameters
        ----------
        batch_shape : Shape
            The shape of the batch to sample. Typically, a tuple indicating the number of samples,
            but an int can also be passed.
        **kwargs
            Additional keyword arguments passed to the sampling function. Only valid arguments
            (as determined by the function's signature) are used.

        Returns
        -------
        data : dict of str to np.ndarray
            A dictionary of sampled outputs. Keys are output names and values are numpy arrays.
            If `is_batched` is False, individual outputs are stacked along the first axis.
        """

        # try to use only valid keyword-arguments
        kwargs = filter_kwargs(kwargs, self.sample_fn)

        if self.is_batched:
            return self.sample_fn(batch_shape, **kwargs)

        data = batched_call(self.sample_fn, batch_shape, kwargs=kwargs, flatten=True)
        data = tree_stack(data, axis=0, numpy=True)

        return data
