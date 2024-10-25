import keras

from bayesflow.data_adapters import DataAdapter
from bayesflow.simulators.simulator import Simulator
from bayesflow.types import Tensor
from bayesflow.utils import logging


class RoundsDataset(keras.utils.PyDataset):
    """
    A dataset that is generated on-the-fly at the beginning of every n-th epoch.
    """

    def __init__(
        self,
        simulator: Simulator,
        batch_size: int,
        num_batches: int,
        epochs_per_round: int,
        data_adapter: DataAdapter | None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        if keras.backend.backend() == "torch" and kwargs.get("use_multiprocessing"):
            # keras workaround: https://github.com/keras-team/keras/issues/19346
            import multiprocessing as mp

            mp.set_start_method("spawn", force=True)

        self.batches = None
        self._num_batches = num_batches
        self.batch_size = batch_size
        self.data_adapter = data_adapter
        self.epoch = 0

        if epochs_per_round == 1:
            logging.warning(
                "Using `RoundsDataset` with `epochs_per_round=1` is equivalent to fully online training. "
                "Use an `OnlineDataset` instead for best performance."
            )

        self.epochs_per_round = epochs_per_round

        self.simulator = simulator

        self.regenerate()

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        """Get a batch of pre-simulated data"""
        batch = self.batches[item]

        if self.data_adapter is not None:
            batch = self.data_adapter(batch, batch_size=self.batch_size)

        return batch

    @property
    def num_batches(self) -> int:
        return self._num_batches

    def on_epoch_end(self) -> None:
        self.epoch += 1
        if self.epoch % self.epochs_per_round == 0:
            self.regenerate()

    def regenerate(self) -> None:
        """Sample new batches of data from the joint distribution unconditionally"""
        self.batches = [self.simulator.sample((self.batch_size,)) for _ in range(self.batches_per_epoch)]
