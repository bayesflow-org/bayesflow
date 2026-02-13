from collections.abc import Callable, Mapping, Sequence

import numpy as np
import keras

from bayesflow.adapters import Adapter
from bayesflow.simulators.simulator import Simulator

from ._augmentations import apply_augmentations


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
        batch = self.simulator.sample(self.batch_size)

        batch = apply_augmentations(batch, self.augmentations)

        if self.adapter is not None:
            batch = self.adapter(batch)

        return batch

    @property
    def num_batches(self) -> int:
        return self._num_batches

    def __len__(self) -> int:
        return self.num_batches
