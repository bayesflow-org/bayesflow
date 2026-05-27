from collections.abc import Sequence

import numpy as np

import keras


class MultiDataset(keras.utils.PyDataset):
    """A dataset that combines multiple named datasets into a single batch.

    On each step, one batch is drawn from every constituent dataset and returned
    together as ``{"data": {"name1": batch1, "name2": batch2, ...}}``. This format
    matches what ``SemiSupervisedApproximator.compute_metrics`` expects.

    When datasets have different lengths, shorter datasets are recycled (modulo
    indexing), so training always runs for ``max(num_batches)`` steps.

    Parameters
    ----------
    **datasets : keras.utils.PyDataset
        Named datasets to combine. Pass as keyword arguments, e.g.
        ``MultiDataset(labeled=ds1, unlabeled=ds2)``.
    """

    def __init__(self, **datasets):
        super().__init__()
        self.datasets = datasets

    def __getitem__(self, item: int) -> Sequence[dict[str, np.ndarray]]:
        """Return one batch from each constituent dataset, wrapping shorter ones."""
        data = {}
        for key, dataset in self.datasets.items():
            data[key] = dataset[item % dataset.num_batches]

        return data

    @property
    def num_batches(self) -> int:
        """Total number of steps per epoch — the maximum across all constituent datasets."""
        num_batches = [dataset.num_batches for dataset in self.datasets.values()]
        return max(num_batches)

    @property
    def dataset_keys(self) -> Sequence[str]:
        """Names of the constituent datasets."""
        return list(self.datasets.keys())
