from collections.abc import Mapping

import numpy as np

import keras

from bayesflow.adapters import Adapter
from bayesflow.utils import logging


class OfflineDataset(keras.utils.PyDataset):
    """
    A dataset that is pre-simulated and stored in memory. When storing and loading data from disk, it is recommended to
    save any pre-simulated data in raw form and create the `OfflineDataset` object only after loading in the raw data.
    See the `DiskDataset` class for handling large datasets that are split into multiple smaller files.
    """

    def __init__(
        self,
        data: Mapping[str, np.ndarray],
        batch_size: int,
        adapter: Adapter | None,
        num_samples: int = None,
        *,
        stage: str = "training",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.data = data
        self.adapter = adapter
        self.stage = stage

        if num_samples is None:
            self.num_samples = self._get_num_samples_from_data(data)
            logging.debug(f"Automatically determined {self.num_samples} samples in data.")
        else:
            self.num_samples = num_samples

        self.indices = np.arange(self.num_samples, dtype="int64")

        self.shuffle()

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        """Get a batch of pre-simulated data"""
        if not 0 <= item < self.num_batches:
            raise IndexError(f"Index {item} is out of bounds for dataset with {self.num_batches} batches.")

        item = slice(item * self.batch_size, (item + 1) * self.batch_size)
        item = self.indices[item]

        batch = {
            key: np.take(value, item, axis=0) if isinstance(value, np.ndarray) else value
            for key, value in self.data.items()
        }

        if self.adapter is not None:
            batch = self.adapter(batch, stage=self.stage)

        return batch

    @property
    def num_batches(self) -> int | None:
        return int(np.ceil(self.num_samples / self.batch_size))

    def on_epoch_end(self) -> None:
        self.shuffle()

    def shuffle(self) -> None:
        """Shuffle the dataset in-place."""
        np.random.shuffle(self.indices)

    @staticmethod
    def _get_num_samples_from_data(data: Mapping) -> int:
        for key, value in data.items():
            if hasattr(value, "shape"):
                ndim = len(value.shape)
                if ndim > 1:
                    return value.shape[0]

        raise ValueError("Could not determine number of samples from data. Please pass it manually.")
