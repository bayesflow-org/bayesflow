import keras
import numpy as np

from bayesflow.data_adapters import DataAdapter


class OfflineDataset(keras.utils.PyDataset):
    """
    A dataset that is pre-simulated and stored in memory.
    """

    def __init__(self, data: dict[str, np.ndarray], batch_size: int, data_adapter: DataAdapter | None, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.data = data
        self.data_adapter = data_adapter
        self.num_samples = next(iter(data.values())).shape[0]
        self.indices = np.arange(self.num_samples, dtype="int64")

        self.shuffle()

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        """Get a batch of pre-simulated data"""
        if not 0 <= item < self.num_batches:
            raise IndexError(f"Index {item} is out of bounds for dataset with {self.num_batches} batches.")

        item = slice(item * self.batch_size, (item + 1) * self.batch_size)
        item = self.indices[item]

        batch = {key: np.take(value, item, axis=0) for key, value in self.data.items()}

        if self.data_adapter is not None:
            batch = self.data_adapter(batch, batch_size=self.batch_size)

        return batch

    @property
    def num_batches(self) -> int | None:
        return int(np.ceil(self.num_samples / self.batch_size))

    def on_epoch_end(self) -> None:
        self.shuffle()

    def shuffle(self) -> None:
        """Shuffle the dataset in-place."""
        np.random.shuffle(self.indices)
