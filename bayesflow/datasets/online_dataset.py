import keras

from bayesflow.simulators.simulator import Simulator
from bayesflow.types import Tensor


class OnlineDataset(keras.utils.PyDataset):
    """
    A dataset that is generated on-the-fly.
    """

    def __init__(self, simulator: Simulator, batch_size: int, **kwargs):
        super().__init__(**kwargs)

        if keras.backend.backend() == "torch" and kwargs.get("use_multiprocessing"):
            # keras workaround: https://github.com/keras-team/keras/issues/19346
            import multiprocessing as mp

            mp.set_start_method("spawn", force=True)

        self.simulator = simulator
        self.batch_size = batch_size

    def __getitem__(self, item: int) -> dict[str, Tensor]:
        return self.simulator.sample((self.batch_size,))

    @property
    def num_batches(self):
        # infinite dataset
        return None
