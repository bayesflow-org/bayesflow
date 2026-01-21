import math
import numpy as np
import keras
from collections.abc import Mapping

from bayesflow.utils import logging

from ._ensemble_sharing import ring_starts, ring_window_indices


class EnsembleIndexedDataset(keras.utils.PyDataset):
    def __init__(
        self,
        dataset: keras.utils.PyDataset,
        num_ensemble: int,
        data_reuse: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_ensemble < 1:
            raise ValueError("EnsembleIndexedDataset: num_ensemble must be >= 1.")
        if not (0.0 <= data_reuse <= 1.0):
            raise ValueError("EnsembleIndexedDataset: data_reuse must be in [0, 1].")

        for attr in ("batch_size", "num_samples", "get_batch_by_sample_indices"):
            if not hasattr(dataset, attr):
                raise TypeError(f"EnsembleIndexedDataset: wrapped dataset must expose `{attr}`.")

        self.dataset = dataset
        self.num_ensemble = int(num_ensemble)
        self.data_reuse = float(data_reuse)
        self.batch_size = int(dataset.batch_size)
        self.num_samples = int(dataset.num_samples)

        self.reduction_factor = 1 / (data_reuse + (1 - data_reuse) * num_ensemble)
        self.window_size = int(math.ceil(self.num_samples * self.reduction_factor))
        self.steps_per_epoch = int(math.ceil(self.window_size / self.batch_size))

        # pool fixed as arange(num_samples)
        pool = np.arange(self.num_samples, dtype="int64")

        starts = ring_starts(self.num_samples, self.num_ensemble)
        idx2d = ring_window_indices(self.num_samples, self.window_size, starts)  # (E, W)
        self.member_indices = [pool[idx2d[m]].copy() for m in range(self.num_ensemble)]

        # initial shuffle of member subdatasets (member_indices)
        self.on_epoch_end()

        logging.info(
            f"EnsembleIndexedDataset: num_ensemble={self.num_ensemble}, "
            f"batch_size={self.batch_size}, num_samples={self.num_samples}, "
            f"data_reuse={self.data_reuse} -> "
            f"reduction_factor={self.reduction_factor:.2f}, window_size={self.window_size}, "
            f"steps_per_epoch={self.steps_per_epoch}. "
            "Overlap is enforced at the subdataset level (member-specific windows into the global index pool)."
        )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def on_epoch_end(self):
        if self.data_reuse == 1.0 or self.num_ensemble == 1:
            np.random.shuffle(self.member_indices[0])
            for m in range(1, self.num_ensemble):
                self.member_indices[m] = self.member_indices[0]
            return

        # otherwise independent shuffle per member
        for m in range(self.num_ensemble):
            np.random.shuffle(self.member_indices[m])

    def __getitem__(self, step: int) -> dict[str, object]:
        if not 0 <= step < self.steps_per_epoch:
            raise IndexError(f"Index {step} is out of bounds for dataset with {self.steps_per_epoch} steps.")

        start = step * self.batch_size
        stop = min((step + 1) * self.batch_size, self.window_size)  # allow shorter last batch

        member_batches = []
        for m in range(self.num_ensemble):
            idx = self.member_indices[m][start:stop]
            member_batches.append(self.dataset.get_batch_by_sample_indices(idx))

        return self._stack_member_batches(member_batches)

    def _stack_member_batches(self, member_batches: list[Mapping[str, object]]) -> dict[str, object]:
        out: dict[str, object] = {}
        keys = member_batches[0].keys()

        for key in keys:
            first = member_batches[0][key]
            if isinstance(first, np.ndarray):
                out[key] = np.stack([mb[key] for mb in member_batches], axis=1)  # (batch, ensemble, ...)
            else:
                out[key] = first
        return out
