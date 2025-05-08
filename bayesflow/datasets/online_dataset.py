from collections.abc import Mapping, Callable

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
        augmentations: Mapping[str, Callable] = None,
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
        augmentations : dict of str to Callable, optional
            Dictionary of augmentation functions to apply to each corresponding key in the batch.
            Note - augmentations are applied before the adapter.
        **kwargs
            Additional keyword arguments passed to the base `PyDataset`.
        """
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self._num_batches = num_batches
        self.adapter = adapter
        self.simulator = simulator
        self.stage = stage
        self.augmentations = augmentations

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

        if self.augmentations is not None:
            for key in self.augmentations:
                batch[key] = self.augmentations[key](batch[key])

        if self.adapter is not None:
            batch = self.adapter(batch, stage=self.stage)

        return batch
