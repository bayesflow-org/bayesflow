from collections.abc import Callable, Mapping, Sequence

import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.simulators.simulator import Simulator


class OnlineDataset(keras.utils.PyDataset):
    """
    A dataset that generates simulations on-the-fly.
    """

    def __init__(
        self,
        simulator: Simulator,
        batch_size: int,
        num_batches: int,
        adapter: Adapter | None,
        *,
        stage: str = "training",
        augmentations: Callable | Mapping[str, Callable] | Sequence[Callable] = None,
        **kwargs,
    ):
        """
        Initialize an OnlineDataset instance for infinite stream training.

        Parameters
        ----------
        simulator : Simulator
            A simulator object with a `.sample(batch_shape)` method to generate data.
        batch_size : int
            Number of samples per batch.
        num_batches : int
            Total number of batches in the dataset.
        adapter : Adapter or None
            Optional adapter to transform the simulated batch.
        stage : str, default="training"
            Current stage (e.g., "training", "validation", etc.) used by the adapter.
        augmentations : Callable or Mapping[str, Callable] or Sequence[Callable], optional
            A single augmentation function, dictionary of augmentation functions, or sequence of augmentation functions
            to apply to the batch.

            If you provide a dictionary of functions, each function should accept one element
            of your output batch and return the corresponding transformed element.

            Otherwise, your function should accept the entire dictionary output and return a dictionary.

            Note - augmentations are applied before the adapter is called and are generally
            transforms that you only want to apply during training.
        **kwargs
            Additional keyword arguments passed to the base `PyDataset`.
        """
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self._num_batches = num_batches
        self.adapter = adapter
        self.simulator = simulator
        self.stage = stage
        self.augmentations = augmentations or []

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        """
        Generate one batch of data.

        Parameters
        ----------
        item : int
            Index of the batch. Required by signature, but not used.

        Returns
        -------
        dict of str to np.ndarray
            A batch of simulated (and optionally augmented/adapted) data.
        """
        batch = self.simulator.sample((self.batch_size,))

        if self.augmentations is None:
            pass
        elif isinstance(self.augmentations, Mapping):
            for key, fn in self.augmentations.items():
                batch[key] = fn(batch[key])
        elif isinstance(self.augmentations, Sequence):
            for fn in self.augmentations:
                batch = fn(batch)
        elif isinstance(self.augmentations, Callable):
            batch = self.augmentations(batch)
        else:
            raise RuntimeError(f"Could not apply augmentations of type {type(self.augmentations)}.")

        if self.adapter is not None:
            batch = self.adapter(batch, stage=self.stage)

        return batch

    @property
    def num_batches(self) -> int:
        return self._num_batches
