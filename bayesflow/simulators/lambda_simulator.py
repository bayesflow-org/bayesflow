from collections.abc import Callable, Sequence

import numpy as np
import keras
from bayesflow.utils import batched_call, filter_kwargs, tree_stack

from .simulator import Simulator


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

    def sample(self, batch_size: int, sample_shape: tuple[int] | None = None, **kwargs) -> dict[str, np.ndarray]:
        """
        Sample using the wrapped sampling function.

        Parameters
        ----------
        batch_size : int
            The number of samples to generate.
        sample_shape : tuple of int or int, optional
            Trailing structural dimensions of each generated sample, excluding the batch and target (intrinsic)
            dimension. For example, if batch_size is `batch_size` and sample_shape is `(time, channels)`, the final
            output will be `(batch_size, time, channels, target_dim)`, where target_dim is the intrinsic dimension of
            the output.

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

        batch_shape = (batch_size, *sample_shape) if sample_shape else (batch_size,)
        if self.is_batched:
            return self.sample_fn(batch_shape, **kwargs)

        data = batched_call(self.sample_fn, batch_shape, kwargs=kwargs, flatten=True)
        data = tree_stack(data, axis=0, numpy=True)
        data = keras.tree.map_structure(lambda x: x.reshape(*batch_shape, -1), data)

        return data
