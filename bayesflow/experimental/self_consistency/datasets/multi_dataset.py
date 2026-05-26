from collections.abc import Sequence

import numpy as np

import keras


class MultiDataset(keras.utils.PyDataset):
    def __init__(self, **datasets):
        super().__init__()
        self.datasets = datasets

    def __getitem__(self, item: int) -> Sequence[dict[str, np.ndarray]]:
        data = {}
        for key, dataset in self.datasets.items():
            num_batches = dataset.num_batches
            item = item % num_batches
            data[key] = dataset[item]

        return dict(data=data)

    @property
    def num_batches(self) -> int:
        num_batches = [dataset.num_batches for dataset in self.datasets.values()]
        return max(num_batches)

    @property
    def dataset_keys(self) -> Sequence[str]:
        return list(self.datasets.keys())
