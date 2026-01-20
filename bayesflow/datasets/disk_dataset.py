from collections.abc import Callable, Mapping, Sequence

import os
import pathlib as pl

import numpy as np

import keras

from bayesflow.adapters import Adapter
from bayesflow.utils import tree_stack, pickle_load


class DiskDataset(keras.utils.PyDataset):
    """
    A dataset used to load pre-simulated files from disk.
    The training strategy will be offline.

    By default, the expected file structure is as follows:
    root
    ├── ...
    ├── sample_1.[ext]
    ├── ...
    └── sample_n.[ext]

    where each file contains a complete sample (e.g., a dictionary of numpy arrays) or
    is converted into a complete sample using a custom loader function.
    """

    def __init__(
        self,
        root: os.PathLike,
        *,
        pattern: str = "*.pkl",
        batch_size: int,
        load_fn: Callable = None,
        adapter: Adapter | None,
        augmentations: Callable | Mapping[str, Callable] | Sequence[Callable] = None,
        shuffle: bool = True,
        **kwargs,
    ):
        """
        Initialize a DiskDataset instance for offline training using a set of simulations that
        do not fit on disk.

        Parameters
        ----------
        root : os.PathLike
            Root directory containing the sample files.
        pattern : str, default="*.pkl"
            Glob pattern to match sample files.
        batch_size : int
            Number of samples per batch.
        load_fn : Callable, optional
            Function to load a single file into a sample. Defaults to `pickle_load`.
        adapter : Adapter or None
            Optional adapter to transform the loaded batch.
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
        self.root = pl.Path(root)
        self.load_fn = load_fn or pickle_load
        self.adapter = adapter
        self.files = list(map(str, self.root.glob(pattern)))

        self.num_samples = len(self.files)
        self.indices = np.arange(self.num_samples, dtype="int64")

        self.augmentations = augmentations or []
        self._shuffle = shuffle
        if self._shuffle:
            self.shuffle()

    def __getitem__(self, item) -> dict[str, np.ndarray]:
        if not 0 <= item < self.num_batches:
            raise IndexError(f"Index {item} is out of bounds for dataset with {self.num_batches} batches.")

        start = item * self.batch_size
        stop = min((item + 1) * self.batch_size, self.num_samples)
        idx = self.indices[start:stop]

        return self.get_batch_by_sample_indices(idx)

    def get_batch_by_sample_indices(self, indices: np.ndarray) -> dict[str, np.ndarray]:
        selected_files = [self.files[int(i)] for i in indices]
        batch = [self.load_fn(file) for file in selected_files]
        batch = tree_stack(batch)

        batch = self._apply_augmentations(batch)

        if self.adapter is not None:
            batch = self.adapter(batch)

        return batch

    def _apply_augmentations(self, batch: dict[str, object]) -> dict[str, object]:
        if self.augmentations is None:
            return batch
        if isinstance(self.augmentations, Mapping):
            for key, fn in self.augmentations.items():
                batch[key] = fn(batch[key])
            return batch
        if isinstance(self.augmentations, Sequence):
            for fn in self.augmentations:
                batch = fn(batch)
            return batch
        if isinstance(self.augmentations, Callable):
            return self.augmentations(batch)
        raise RuntimeError(f"Could not apply augmentations of type {type(self.augmentations)}.")

    def on_epoch_end(self):
        if self._shuffle:
            self.shuffle()

    @property
    def num_batches(self):
        return int(np.ceil(self.num_samples / self.batch_size))

    def __len__(self) -> int:
        return self.num_batches

    def shuffle(self):
        np.random.shuffle(self.indices)
