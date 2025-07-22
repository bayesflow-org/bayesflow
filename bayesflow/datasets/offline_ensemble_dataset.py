import numpy as np

from .offline_dataset import OfflineDataset


class OfflineEnsembleDataset(OfflineDataset):
    """
    A dataset that is pre-simulated and stored in memory, extending :py:class:`OfflineDataset`.

    The only difference is that it allows to train an :py:class:`ApproximatorEnsemble` in parallel by returning
    batches with ``num_ensemble`` different random subsets of the available data.
    """

    def __init__(self, num_ensemble: int, **kwargs):
        super().__init__(**kwargs)
        self.num_ensemble = num_ensemble

        # Create indices with shape (num_samples, num_ensemble)
        _indices = np.arange(self.num_samples, dtype="int64")
        _indices = np.repeat(_indices[:, None], self.num_ensemble, axis=1)

        # Shuffle independently along second axis
        for i in range(self.num_ensemble):
            np.random.shuffle(_indices[:, i])

        self.indices = _indices

        # Shuffle first axis
        if self._shuffle:
            self.shuffle()
