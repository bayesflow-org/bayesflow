import math
from collections.abc import Callable, Mapping, Sequence
import numpy as np
import keras


from bayesflow.utils import logging

from ._ensemble_sharing import ring_starts, ring_window_indices


class EnsembleOnlineDataset(keras.utils.PyDataset):
    def __init__(
        self,
        dataset: keras.utils.PyDataset,
        num_ensemble: int,
        data_reuse: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_ensemble < 1:
            raise ValueError("EnsembleOnlineDataset: num_ensemble must be >= 1.")
        if not (0.0 <= data_reuse <= 1.0):
            raise ValueError("EnsembleOnlineDataset: data_reuse must be in [0, 1].")

        for attr in ("simulator", "batch_size", "num_batches", "augmentations", "adapter"):
            if not hasattr(dataset, attr):
                raise TypeError(f"EnsembleOnlineDataset: wrapped dataset must expose `{attr}`.")

        self.dataset = dataset
        self.num_ensemble = num_ensemble
        self.data_reuse = data_reuse
        self.batch_size = int(dataset.batch_size)
        self._num_batches = dataset.num_batches

        self.reduction_factor = 1 / (data_reuse + (1 - data_reuse) * num_ensemble)
        self.pool_size = int(math.ceil(self.batch_size / self.reduction_factor))

        logging.info(
            f"EnsembleOnlineDataset: num_ensemble={self.num_ensemble}, "
            f"batch_size={self.batch_size}, data_reuse={self.data_reuse} -> "
            f"reduction_factor={self.reduction_factor:.2f}, "
            f"pool_size={self.pool_size} (≈{1 / self.reduction_factor:.1f}*batch_size). "
            "Overlap is enforced per training step by splitting a pooled simulated batch into member windows."
        )

    @property
    def num_batches(self):
        return self._num_batches

    def __len__(self) -> int:
        return self.num_batches

    def __getitem__(self, item: int) -> dict[str, object]:
        # fully shared
        if self.data_reuse == 1.0 or self.num_ensemble == 1:
            batch = self.dataset.simulator.sample((self.batch_size,))
            batch = self._postprocess(batch)
            return self._replicate(batch)

        pool = self.dataset.simulator.sample((self.pool_size,))

        pool = self._apply_augmentations(pool)
        if self.dataset.adapter is not None:
            batch = self.dataset.adapter(batch)

        starts = ring_starts(self.pool_size, self.num_ensemble)
        idx2d = ring_window_indices(self.pool_size, self.batch_size, starts)

        member_batches = []
        for m in range(self.num_ensemble):
            member_batches.append(self._take(pool, idx2d[m]))

        return self._stack(member_batches)

    def _apply_augmentations(self, batch: dict[str, object]) -> dict[str, object]:
        aug = self.dataset.augmentations

        if aug is None:
            return batch
        if isinstance(aug, Mapping):
            for key, fn in aug.items():
                batch[key] = fn(batch[key])
            return batch
        if isinstance(aug, Sequence):
            for fn in aug:
                batch = fn(batch)
            return batch
        if isinstance(aug, Callable):
            return aug(batch)
        raise RuntimeError(f"Could not apply augmentations of type {type(aug)}.")

    @staticmethod
    def _take(batch: Mapping[str, object], idx: np.ndarray) -> dict[str, object]:
        out = {}
        for k, v in batch.items():
            out[k] = np.take(v, idx, axis=0) if isinstance(v, np.ndarray) else v
        return out

    def _replicate(self, batch: Mapping[str, object]) -> dict[str, object]:
        out = {}
        for k, v in batch.items():
            if isinstance(v, np.ndarray):
                out[k] = np.repeat(np.expand_dims(v, 1), self.num_ensemble, axis=1)
            else:
                out[k] = v
        return out

    @staticmethod
    def _stack(member_batches: list[Mapping[str, object]]) -> dict[str, object]:
        out = {}
        keys = member_batches[0].keys()
        for k in keys:
            v0 = member_batches[0][k]
            out[k] = np.stack([mb[k] for mb in member_batches], axis=1) if isinstance(v0, np.ndarray) else v0
        return out
