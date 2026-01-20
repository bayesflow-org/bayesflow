import math
from collections.abc import Mapping

import keras
import numpy as np

from bayesflow.utils import logging


class EnsembleDatasetWrapper(keras.utils.PyDataset):
    def __init__(
        self,
        dataset: keras.utils.PyDataset,
        num_ensemble: int,
        batch_independence: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if num_ensemble < 1:
            raise ValueError("EnsembleDatasetWrapper: num_ensemble must be >= 1.")
        if not (0.0 <= batch_independence <= 1.0):
            raise ValueError("EnsembleDatasetWrapper: batch_independence must be in [0, 1].")

        self.dataset = dataset
        self.num_ensemble = int(num_ensemble)
        self.batch_independence = float(batch_independence)

        # Determine batch_size from the wrapped dataset if possible, else infer from a probe batch.
        if hasattr(self.dataset, "batch_size"):
            self.batch_size = int(getattr(self.dataset, "batch_size"))
        else:
            probe_batch = self.dataset[0]
            self.batch_size = self._infer_num_samples_from_batch(probe_batch)

        # Determine num_batches for safe modulo indexing.
        self._num_batches = getattr(self.dataset, "num_batches", None)
        if self._num_batches is None and hasattr(self.dataset, "__len__"):
            self._num_batches = len(self.dataset)
        if self._num_batches is None:
            raise ValueError(
                "EnsembleDatasetWrapper: wrapped dataset must expose `num_batches` or implement `__len__`."
            )
        self._num_batches = int(self._num_batches)

        # Precompute pooling schedule (even if not used in the endpoint fast paths).
        self.pool_num_batches_per_step = int(
            math.ceil((1.0 - self.batch_independence) + self.num_ensemble * self.batch_independence)
        )
        self.pool_num_batches_per_step = max(1, self.pool_num_batches_per_step)
        self.pool_size_per_step = self.pool_num_batches_per_step * self.batch_size

        match self.batch_independence:
            case 0.0:
                logging.info(
                    "EnsembleDatasetWrapper: batch_independence=0.0, num_ensemble=%d, batch_size=%d -> "
                    "one batch is drawn per training step and replicated for all ensemble members.",
                    self.num_ensemble,
                    self.batch_size,
                )

            case 1.0:
                logging.info(
                    "EnsembleDatasetWrapper: batch_independence=1.0, num_ensemble=%d, batch_size=%d -> "
                    "num_ensemble=%d batches are drawn per training step (one per ensemble member, no pooling).",
                    self.num_ensemble,
                    self.batch_size,
                    self.num_ensemble,
                )

            case _:
                logging.info(
                    "EnsembleDatasetWrapper: batch_independence=%s, num_ensemble=%d, batch_size=%d -> "
                    "%d wrapped-dataset batches per step are used as a pool "
                    "(pool_size=%d = %d×batch_size); ensemble members receive deterministic overlapping slices.",
                    self.batch_independence,
                    self.num_ensemble,
                    self.batch_size,
                    self.pool_num_batches_per_step,
                    self.pool_size_per_step,
                    self.pool_num_batches_per_step,
                )

    @property
    def num_batches(self) -> int:
        return self._num_batches

    # TODO: QUESTION FOR CODE REVIEW: currently all our datasets use a @property for num_batches
    # instead of a simple attribute.
    # For OfflineDataset and DiskDataset it is actually computed on access,
    # while OnlineDataset just relays the internal attribute _num_batches.
    # Consistency: always @property
    #  vs.
    # Simplicity: use only simple attribute when no computation on access takes place.
    # I am going with consistency for now, it's not too important. Public API doesn't change in either case.
    # If we choose to favor simplicity, we should also change OnlineDataset accordingly.

    def __len__(self) -> int:
        return self.num_batches

    # TODO: QUESTION FOR CODE REVIEW: we could always support len(dataset) ?

    def on_epoch_end(self):
        if hasattr(self.dataset, "on_epoch_end"):
            self.dataset.on_epoch_end()

    def __getitem__(self, item: int) -> dict[str, object]:
        # Special case: all ensemble members receive the same batch
        if self.batch_independence == 0.0 or self.num_ensemble == 1:
            base_batch = self.dataset[int(item) % self.num_batches]
            return self._replicate_across_ensemble_axis(base_batch)

        # Special case: each ensemble member receives its own batch (no pooling)
        if self.batch_independence == 1.0:
            member_batches = self._collect_member_batches_no_pool(item)
            return self._stack_member_batches(member_batches)

        # General case: pooling with whole wrapped-dataset batches, ensemble members recieve overlapping windows of pool
        pool = self._collect_pool(item)

        member_batches = []

        # Last member gets batch that starts at pool_size_per_step - batch_size
        max_start = self._infer_num_samples_from_batch(pool) - self.batch_size

        # Spread start positions deterministically from 0 (member 0) to max_start (last member).
        start_positions = [
            int((member_index * max_start) // (self.num_ensemble - 1)) for member_index in range(self.num_ensemble)
        ]

        for start in start_positions:
            member_batches.append(self._slice_from_batch(pool, start, start + self.batch_size))

        return self._stack_member_batches(member_batches)

    # -------------------------
    # Helper methods
    # -------------------------

    def _infer_num_samples_from_batch(self, batch: Mapping[str, object]) -> int:
        for value in batch.values():
            if isinstance(value, np.ndarray) and value.ndim >= 1:
                return int(value.shape[0])
        raise ValueError(
            "EnsembleDatasetWrapper: could not infer batch_size from wrapped dataset output. "
            "Expected at least one np.ndarray with a leading sample dimension."
        )

    def _collect_pool(self, item: int) -> dict[str, object]:
        batches = []
        base_index = int(item) * self.pool_num_batches_per_step
        for offset in range(self.pool_num_batches_per_step):
            wrapped_index = (base_index + offset) % self.num_batches
            batches.append(self.dataset[wrapped_index])
        return self._concat_batches(batches)

    def _collect_member_batches_no_pool(self, item: int) -> list[dict[str, object]]:
        member_batches = []
        base_index = int(item) * self.num_ensemble
        for member_index in range(self.num_ensemble):
            wrapped_index = (base_index + member_index) % self.num_batches
            member_batch = self.dataset[wrapped_index]
            # Ensure member batch has correct batch_size. Not in general the case for final OfflineDataset batch.
            member_batches.append(self._ensure_full_batch_size(member_batch))
        return member_batches

    def _ensure_full_batch_size(self, batch: dict[str, object]) -> dict[str, object]:
        batch_num_samples = self._infer_batch_size_from_batch(batch)
        if batch_num_samples == self.batch_size:
            return batch

        indices = np.arange(self.batch_size) % batch_num_samples
        return self._take_from_batch(batch, indices)

    def _concat_batches(self, batches: list[Mapping[str, object]]) -> dict[str, object]:
        if not batches:
            raise ValueError("EnsembleDatasetWrapper: cannot concatenate an empty list of batches.")

        out: dict[str, object] = {}
        keys = batches[0].keys()
        for key in keys:
            first_value = batches[0][key]
            if isinstance(first_value, np.ndarray):
                out[key] = np.concatenate([b[key] for b in batches], axis=0)
            else:
                # Non-array metadata (rare): keep the first.
                out[key] = first_value
        return out

    def _take_from_batch(self, batch: Mapping[str, object], indices: np.ndarray) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                out[key] = np.take(value, indices, axis=0)
            else:
                out[key] = value
        return out

    def _slice_from_batch(self, batch: Mapping[str, object], start: int, stop: int) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                out[key] = value[start:stop]
            else:
                out[key] = value
        return out

    def _replicate_across_ensemble_axis(self, batch: Mapping[str, object]) -> dict[str, object]:
        out: dict[str, object] = {}
        for key, value in batch.items():
            if isinstance(value, np.ndarray):
                expanded = np.expand_dims(value, axis=1)  # (batch_size, 1, ...)
                out[key] = np.repeat(expanded, self.num_ensemble, axis=1)  # (batch_size, num_ensemble, ...)
            else:
                out[key] = value
        return out

    def _stack_member_batches(self, member_batches: list[Mapping[str, object]]) -> dict[str, object]:
        if len(member_batches) != self.num_ensemble:
            raise ValueError(
                f"EnsembleDatasetWrapper: expected exactly num_ensemble member batches, got {len(member_batches)}."
            )

        out: dict[str, object] = {}
        keys = member_batches[0].keys()

        for key in keys:
            first_value = member_batches[0][key]
            if isinstance(first_value, np.ndarray):
                # Stack along axis=1 -> (batch_size, num_ensemble, ...)
                out[key] = np.stack([mb[key] for mb in member_batches], axis=1)
            else:
                out[key] = first_value
        return out
