from collections.abc import Callable, Mapping, Sequence

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
        augmentations: Callable | Mapping[str, Callable] | Sequence[Callable] = None,
        shuffle: bool = True,
        **kwargs,
    ):
        """
        Initialize an OfflineDataset instance for offline training with optional data augmentations.

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            Pre-simulated data stored in a dictionary, where each key maps to a NumPy array.
        batch_size : int
            Number of samples per batch.
        adapter : Adapter or None
            Optional adapter to transform the batch.
        num_samples : int, optional
            Number of samples in the dataset. If None, it will be inferred from the data.
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
        shuffle : bool, optional
            Whether to shuffle the dataset at initialization and at the end of each epoch. Default is True.
        **kwargs
            Additional keyword arguments passed to the base `PyDataset`.
        """
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

        self.augmentations = augmentations or []
        self._shuffle = shuffle
        if self._shuffle:
            self.shuffle()

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        """
        Load a batch of data from disk.

        Parameters
        ----------
        item : int
            Index of the batch to retrieve.

        Returns
        -------
        dict of str to np.ndarray
            A batch of loaded (and optionally augmented/adapted) data.

        Raises
        ------
        IndexError
            If the requested batch index is out of range.
        """
        if not 0 <= item < self.num_batches:
            raise IndexError(f"Index {item} is out of bounds for dataset with {self.num_batches} batches.")

        item = slice(item * self.batch_size, (item + 1) * self.batch_size)
        item = self.indices[item]

        batch = {
            key: np.take(value, item, axis=0) if isinstance(value, np.ndarray) else value
            for key, value in self.data.items()
        }

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
    def num_batches(self) -> int | None:
        return int(np.ceil(self.num_samples / self.batch_size))

    def on_epoch_end(self) -> None:
        if self._shuffle:
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
