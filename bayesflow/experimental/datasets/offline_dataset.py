
import keras

from bayesflow.experimental.utils import nested_getitem


class OfflineDataset(keras.utils.PyDataset):
    """
    A dataset that is pre-simulated and stored in memory.
    """
    # TODO: fix
    def __init__(self, data: dict, batch_size: int, batches_per_epoch: int, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.batches_per_epoch = batches_per_epoch

        self.data = data
        self.indices = keras.ops.arange(batch_size * batches_per_epoch)

        self.shuffle()

    def __getitem__(self, item: int) -> (dict, dict):
        """ Get a batch of pre-simulated data """
        item = slice(item * self.batch_size, (item + 1) * self.batch_size)
        item = self.indices[item]
        data = nested_getitem(self.data, item)
        return data, {}

    def __len__(self) -> int:
        return self.batches_per_epoch

    def on_epoch_end(self) -> None:
        self.shuffle()

    def shuffle(self) -> None:
        """ Shuffle the dataset in-place. """
        self.indices = keras.random.shuffle(self.indices)
