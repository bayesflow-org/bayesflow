import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.simulators.simulator import Simulator


class OnlineDataset(keras.utils.PyDataset):
    """
    A dataset that is generated on-the-fly.
    """

    def __init__(
        self,
        simulator: Simulator,
        batch_size: int,
        num_batches: int,
        adapter: Adapter | None,
        *,
        stage: str = "training",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.batch_size = batch_size
        self._num_batches = num_batches
        self.adapter = adapter
        self.simulator = simulator
        self.stage = stage

    def __getitem__(self, item: int) -> dict[str, np.ndarray]:
        batch = self.simulator.sample((self.batch_size,))

        if self.adapter is not None:
            batch = self.adapter(batch, stage=self.stage)

        return batch

    @property
    def num_batches(self) -> int:
        return self._num_batches
