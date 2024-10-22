from collections.abc import Mapping
import numpy as np

from bayesflow.utils import batched_call, filter_kwargs, tree_stack

from .simulator import Simulator
from bayesflow.utils import validate_shape
from ..types import Shape, ShapeLike


class LambdaSimulator(Simulator):
    def __init__(
        self,
        sample_fn: callable,
        *,
        is_batched: bool = False,
        cast_dtypes: Mapping[str, str] = "default",
        reserved_arguments: Mapping[str, any] = "default",
    ):
        """Implements a simulator based on a (batched or unbatched) sampling function.
        Outputs will always be in batched format.
        :param sample_fn: The sampling function.
            If in batched format, must accept a batch_shape argument as the first positional argument.
            If in unbatched format (the default), may accept any keyword arguments.
            Must return a dictionary of string keys and numpy array (or scalar) values.
        :param is_batched: Whether the sampling function is in batched format.
        :param cast_dtypes: Output data types to cast arrays to.
            By default, we convert float64 (the default for numpy on x64 systems)
            to float32 (the default for deep learning on any system).
        :param reserved_arguments: Reserved keyword arguments to pass to the sampling function.
            By default, functions requesting an argument 'rng' will be passed the default numpy random generator.
        """
        self.sample_fn = sample_fn
        self.is_batched = is_batched

        if cast_dtypes == "default":
            cast_dtypes = {"float64": "float32"}

        self.cast_dtypes = cast_dtypes

        if reserved_arguments == "default":
            reserved_arguments = {"rng": np.random.default_rng()}

        self.reserved_arguments = reserved_arguments

    def sample(self, batch_shape: ShapeLike, **kwargs) -> dict[str, np.ndarray]:
        batch_shape = validate_shape(batch_shape)

        # add reserved arguments
        kwargs = self.reserved_arguments | kwargs

        # try to use only valid keyword arguments
        kwargs = filter_kwargs(kwargs, self.sample_fn)

        if self.is_batched:
            data = self.sample_fn(batch_shape, **kwargs)
        else:
            data = self._sample_batch(batch_shape, **kwargs)

        data = self._cast_dtypes(data)

        return data

    def _sample_batch(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        """Samples a batch of data from an otherwise unbatched sampling function."""
        data = batched_call(self.sample_fn, batch_shape, kwargs=kwargs, flatten=True)

        data = tree_stack(data, axis=0, numpy=True)

        return data

    def _cast_dtypes(self, data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        data = data.copy()

        for key, value in data.items():
            dtype = str(value.dtype)
            if dtype in self.cast_dtypes:
                data[key] = value.astype(self.cast_dtypes[dtype])

        return data
