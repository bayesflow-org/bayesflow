import keras
import numpy as np
import os
import pathlib as pl

from bayesflow.data_adapters import DataAdapter
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
        load_fn: callable = None,
        data_adapter: DataAdapter | None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.root = pl.Path(root)
        self.load_fn = load_fn or pickle_load
        self.data_adapter = data_adapter
        self.files = list(map(str, self.root.glob(pattern)))

        self.shuffle()

    def __getitem__(self, item):
        if not 0 <= item < self.num_batches:
            raise IndexError(f"Index {item} is out of bounds for dataset with {self.num_batches} batches.")

        files = self.files[item * self.batch_size : (item + 1) * self.batch_size]

        batch = []
        for file in files:
            batch.append(self.load_fn(file))

        batch = tree_stack(batch)

        if self.data_adapter is not None:
            batch = self.data_adapter(batch, batch_size=self.batch_size)

        return batch

    def on_epoch_end(self):
        self.shuffle()

    @property
    def num_batches(self):
        return int(np.ceil(len(self.files) / self.batch_size))

    def shuffle(self):
        np.random.shuffle(self.files)
