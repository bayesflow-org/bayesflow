from collections.abc import Sequence
from typing import Any

import math
import numpy as np
import keras

from bayesflow.utils import logging

from ._ensemble_sharing import ring_starts, ring_window_indices


class EnsembleIndexedDataset(keras.utils.PyDataset):
    def __init__(
        self,
        dataset: keras.utils.PyDataset,
        member_names: Sequence[str],
        data_reuse: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if len(member_names) < 2:
            raise ValueError("EnsembleIndexedDataset: len(member_names) must be >= 2.")
        if not (0.0 <= data_reuse <= 1.0):
            raise ValueError("EnsembleIndexedDataset: data_reuse must be in [0, 1].")

        for attr in ("batch_size", "num_samples", "get_batch_by_sample_indices"):
            if not hasattr(dataset, attr):
                raise TypeError(f"EnsembleIndexedDataset: wrapped dataset must expose `{attr}`.")

        self.dataset = dataset
        self.member_names = list(member_names)
        self.ensemble_size = len(member_names)
        self.data_reuse = float(data_reuse)
        self.batch_size = int(dataset.batch_size)
        self.num_samples = int(dataset.num_samples)

        self.reduction_factor = 1 / (data_reuse + (1 - data_reuse) * self.ensemble_size)
        self.window_size = int(math.ceil(self.num_samples * self.reduction_factor))
        self.steps_per_epoch = int(math.ceil(self.window_size / self.batch_size))

        # pool fixed as arange(num_samples)
        pool = np.arange(self.num_samples, dtype="int64")

        starts = ring_starts(self.num_samples, self.ensemble_size)
        idx2d = ring_window_indices(self.num_samples, self.window_size, starts)  # (E, W)
        self.member_indices = {name: pool[idx2d[k]].copy() for k, name in enumerate(self.member_names)}

        # initial shuffle of member subdatasets (member_indices)
        self.on_epoch_end()

        logging.info(
            f"EnsembleIndexedDataset: ensemble_size={self.ensemble_size}, "
            f"batch_size={self.batch_size}, num_samples={self.num_samples}, "
            f"data_reuse={self.data_reuse} -> "
            f"reduction_factor={self.reduction_factor:.2f}, window_size={self.window_size}, "
            f"steps_per_epoch={self.steps_per_epoch}. "
            "Overlap is enforced at the subdataset level (member-specific windows into the global index pool)."
        )

    def __len__(self) -> int:
        return self.steps_per_epoch

    def on_epoch_end(self):
        if self.data_reuse == 1.0:
            np.random.shuffle(self.member_indices[self.member_names[0]])
            for name in self.member_names[1:]:
                self.member_indices[name] = self.member_indices[self.member_names[0]]
            return

        # otherwise independent shuffle per member
        for name in self.member_names:
            np.random.shuffle(self.member_indices[name])

    def __getitem__(self, step: int) -> dict[str, dict[str, Any]]:
        if not 0 <= step < self.steps_per_epoch:
            raise IndexError(f"Index {step} is out of bounds for dataset with {self.steps_per_epoch} steps.")

        start = step * self.batch_size
        stop = min((step + 1) * self.batch_size, self.window_size)  # allow shorter last batch

        out: dict[str, dict[str, Any]] = {}
        for name in self.member_names:
            idx = self.member_indices[name][start:stop]
            out[name] = self.dataset.get_batch_by_sample_indices(idx)

        return self._flip_nested_dict(out)

    def _flip_nested_dict(self, d: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        flipped = {}
        for key, val in d.items():
            for subkey, subval in val.items():
                flipped.setdefault(subkey, {})
                flipped[subkey][key] = subval
        return flipped
