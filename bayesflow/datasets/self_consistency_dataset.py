import keras

from .multi_dataset import MultiDataset


class SelfConsistencyDataset(MultiDataset):
    def __init__(
        self,
        labeled: keras.utils.PyDataset = None,
        unlabeled: keras.utils.PyDataset = None,
        **kwargs,
    ):
        super().__init__(labeled=labeled, unlabeled=unlabeled, **kwargs)
