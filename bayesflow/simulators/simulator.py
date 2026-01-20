from collections.abc import Callable
import numpy as np
from bayesflow.types import Shape
from bayesflow.utils import tree_concatenate
from bayesflow.utils.decorators import allow_batch_size


class Simulator:
    def sample(self, batch_size: int, sample_shape: tuple[int] | None = None, **kwargs) -> dict[str, np.ndarray]:
        raise NotImplementedError

    def rejection_sample(
        self,
        batch_size: int,
        predicate: Callable[[dict[str, np.ndarray]], np.ndarray],
        *,
        sample_shape: tuple[int] | None = None,
        axis: int = 0,
        sample_size: int | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        batch_shape = (batch_size, *sample_shape) if sample_shape else (batch_size,)

        if sample_size is None:
            iteration_batch_shape = batch_shape
        else:
            iteration_batch_shape = list(batch_shape)
            iteration_batch_shape[axis] = sample_size

            iteration_batch_shape = tuple(iteration_batch_shape)

        result = {}

        while not result or next(iter(result.values())).shape[axis] < batch_shape[axis]:
            # get a batch of samples
            iteration_batch_size = iteration_batch_shape[0]
            iteration_sample_shape = iteration_batch_shape[1:] if len(iteration_batch_shape) > 1 else None
            samples = self.sample(iteration_batch_size, sample_shape=iteration_sample_shape, **kwargs)

            # get acceptance mask and turn into indices
            accept = predicate(samples)

            if not isinstance(accept, np.ndarray):
                raise RuntimeError("Predicate must return a numpy array.")

            if accept.shape != (iteration_batch_shape[axis],):
                raise RuntimeError(
                    f"Predicate return array must have shape {(iteration_batch_shape[axis],)}. Received: {accept.shape}."
                )

            if not accept.dtype == "bool":
                # we could cast, but this tends to hide mistakes in the predicate
                raise RuntimeError(f"Predicate must return a boolean type array. Got dtype={accept.dtype}")

            if not np.any(accept):
                # no samples accepted, skip
                continue

            (accept,) = np.nonzero(accept)

            # apply acceptance mask
            samples = {key: np.take(value, accept, axis=axis) for key, value in samples.items()}

            # concatenate with previous samples
            if not result:
                result = samples
            else:
                result = tree_concatenate([result, samples], axis=axis, numpy=True)

        return result

    def sample_batched(
        self,
        batch_size: int,
        *,
        sample_shape: tuple[int] | None = None,
        sample_size: int,
        **kwargs,
    ):
        """Sample the desired number of simulations in smaller batches.

        Limited resources, especially memory, can make it necessary to run simulations in smaller batches.
        The number of samples per simulated batch is specified by `sample_size`.

        Parameters
        ----------
        batch_shape : Shape
            The desired output shape, as in :py:meth:`sample`. Will be rounded up to the next complete batch.
        sample_size : int
            The number of samples in each simulated batch.
        kwargs
            Additional keyword arguments passed to :py:meth:`sample`.

        """

        def accept_all_predicate(x):
            return np.full((sample_size,), True)

        return self.rejection_sample(
            batch_size, sample_shape=sample_shape, predicate=accept_all_predicate, sample_size=sample_size, **kwargs
        )
