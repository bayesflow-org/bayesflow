import keras

from .ensemble_online_dataset import EnsembleOnlineDataset
from .ensemble_indexed_dataset import EnsembleIndexedDataset


class EnsembleDataset(keras.utils.PyDataset):
    """
    Wrap a BayesFlow dataset to provide per-ensemble-member batches.

    This dataset class is the recommended entry point for training ensembles.
    The wrapped dataset should meet the requirements of any single approximator in
    the :class:`~bayesflow.approximators.ApproximatorEnsemble`. It works with
    :class:`~bayesflow.datasets.OnlineDataset`, :class:`~bayesflow.datasets.OfflineDataset`,
    and :class:`~bayesflow.datasets.DiskDataset` and returns batches whose array entries
    have shape ``(batch, ensemble, ...)``.

    The wrapper controls how much data is shared between ensemble members through the
    ``data_reuse`` parameter:

    - ``data_reuse = 1.0``: all ensemble members receive identical data.
    - ``data_reuse = 0.0``: each member receives maximally different data.
    - intermediate values: the total amount of data used per step / per epoch interpolates
      linearly between these extremes.

    Notes
    -----
    Implementation details differ by dataset type:

    **OnlineDataset**
        A larger "pool" of simulations is generated per training step and split into
        overlapping member batches (sharing is enforced per batch).
        This is implemented by :class:`~bayesflow.datasets.EnsembleOnlineDataset`.

    **OfflineDataset / DiskDataset**
        A member-specific subdataset (window into the full index set) is constructed once
        on initialization. Batches are drawn from these subdatasets and reshuffled on
        ``on_epoch_end`` (sharing is enforced at the subdataset level).
        This is implemented by :class:`~bayesflow.datasets.EnsembleIndexedDataset`.

    Parameters
    ----------
    dataset : keras.utils.PyDataset
        A BayesFlow dataset (OnlineDataset, OfflineDataset, DiskDataset).
    num_ensemble : int
        Number of ensemble members.
    data_reuse : float, default=1.0
        Degree of independence between ensemble members in ``[0, 1]``.
        See Notes for how it is applied for different dataset types.
    """

    def __init__(
        self,
        dataset: keras.utils.PyDataset,
        num_ensemble: int,
        data_reuse: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Dispatch based on capabilities (duck typing)
        if hasattr(dataset, "simulator") and hasattr(dataset, "num_batches"):
            self._impl = EnsembleOnlineDataset(
                dataset,
                num_ensemble=num_ensemble,
                data_reuse=data_reuse,
            )
        elif hasattr(dataset, "get_batch_by_sample_indices") and hasattr(dataset, "num_samples"):
            self._impl = EnsembleIndexedDataset(
                dataset,
                num_ensemble=num_ensemble,
                data_reuse=data_reuse,
            )
        else:
            raise TypeError(
                "EnsembleDataset: dataset must be OnlineDataset-like (has `.simulator`) "
                "or Offline/Disk-like (has `num_samples` and `get_batch_by_sample_indices`)."
            )

    def __len__(self) -> int:
        return len(self._impl)

    def __getitem__(self, item: int) -> dict[str, object]:
        return self._impl[item]

    def on_epoch_end(self):
        if hasattr(self._impl, "on_epoch_end"):
            self._impl.on_epoch_end()

    @property
    def num_batches(self) -> int:
        # provide a consistent attribute if the impl has it, else fall back to __len__
        return int(getattr(self._impl, "num_batches", len(self._impl)))
