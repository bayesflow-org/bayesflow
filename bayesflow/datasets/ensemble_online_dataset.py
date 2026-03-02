from collections.abc import Mapping, Sequence
from typing import Any

import math
import numpy as np
import keras

from bayesflow.utils import logging

from ._augmentations import apply_augmentations
from ._ensemble_sharing import ring_starts, ring_window_indices


class EnsembleOnlineDataset(keras.utils.PyDataset):
    def __init__(
        self,
        dataset: keras.utils.PyDataset,
        member_names: Sequence[str],
        data_reuse: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if len(member_names) < 2:
            raise ValueError("EnsembleOnlineDataset: len(member_names) must be >= 2.")
        if not (0.0 <= data_reuse <= 1.0):
            raise ValueError("EnsembleOnlineDataset: data_reuse must be in [0, 1].")

        for attr in ("simulator", "batch_size", "num_batches", "augmentations", "adapter"):
            if not hasattr(dataset, attr):
                raise TypeError(f"EnsembleOnlineDataset: wrapped dataset must expose `{attr}`.")

        self.dataset = dataset
        self.member_names = list(member_names)
        self.ensemble_size = len(self.member_names)
        self.data_reuse = data_reuse
        self.batch_size = int(dataset.batch_size)
        self._num_batches = dataset.num_batches

        self.reduction_factor = 1 / (data_reuse + (1 - data_reuse) * self.ensemble_size)
        self.pool_size = int(math.ceil(self.batch_size / self.reduction_factor))

        logging.info(
            f"EnsembleOnlineDataset: ensemble_size={self.ensemble_size}, "
            f"batch_size={self.batch_size}, data_reuse={self.data_reuse} -> "
            f"reduction_factor={self.reduction_factor:.1f}, "
            f"pool_size={self.pool_size} (≈{1 / self.reduction_factor:.1f}*batch_size).\n"
            "Overlap is enforced per training step by splitting a pooled simulated batch into member windows."
        )

    @property
    def num_batches(self):
        return self._num_batches

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, item: int) -> dict[str, dict[str, Any]]:
        if self.data_reuse == 1.0:
            batch = self.dataset.simulator.sample(self.batch_size)
            batch = self._postprocess(batch)
            return self._replicate(batch)

        pool = self.dataset.simulator.sample(self.pool_size)

        pool = self._postprocess(pool)

        starts = ring_starts(self.pool_size, self.ensemble_size)
        idx2d = ring_window_indices(self.pool_size, self.batch_size, starts)

        out = {}
        for k, name in enumerate(self.member_names):
            out[name] = self._take(pool, idx2d[k])

        return self._flip_nested_dict(out)

    def _flip_nested_dict(self, d: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
        flipped = {}
        for key, val in d.items():
            for subkey, subval in val.items():
                flipped.setdefault(subkey, {})
                flipped[subkey][key] = subval
        return flipped

    def _postprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        batch = apply_augmentations(batch, self.dataset.augmentations)
        if self.dataset.adapter is not None:
            batch = self.dataset.adapter(batch)
        return batch

    @staticmethod
    def _take(batch: Mapping[str, Any], idx: np.ndarray) -> dict[str, Any]:
        out = {}
        for k, v in batch.items():
            out[k] = np.take(v, idx, axis=0) if isinstance(v, np.ndarray) else v
        return out

    def _replicate(self, batch: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        out = {}
        for key, value in batch.items():
            out[key] = {}
            for name in self.member_names:
                out[key][name] = value
        return out
