from collections.abc import Callable, Mapping, Sequence
from bayesflow.utils import logging
import keras
import numpy as np

from bayesflow.adapters import Adapter

from ..graphical_simulator import GraphicalSimulator
from .utils import inference_variables_by_network, inference_conditions_by_network, summary_inputs_by_network


class GraphicalDataset(keras.utils.PyDataset):
    """
    A dataset that generates simulations on-the-fly.
    """

    def __init__(
        self,
        *,
        dataset=None,
        simulator=None,
        approximator=None,
        batch_size=None,
        num_batches=None,
        num_samples=None,
        adapter=None,
        augmentations: Callable | Mapping[str, Callable] | Sequence[Callable] | None = None,
        **kwargs,
    ):
        """
        Initialize an OnlineDataset instance for infinite stream training.

        Parameters
        ----------
        simulator : Simulator
            A simulator object with a `.sample(batch_shape)` method to generate data.
        batch_size : int
            Number of samples per batch.
        num_batches : int
            Total number of batches in the dataset.
        adapter : Adapter or None
            Optional adapter to transform the simulated batch.
        augmentations : Callable or Mapping[str, Callable] or Sequence[Callable], optional
            A single augmentation function, dictionary of augmentation functions, or sequence of augmentation functions
            to apply to the batch.

            If you provide a dictionary of functions, each function should accept one element
            of your output batch and return the corresponding transformed element.

            Otherwise, your function should accept the entire dictionary output and return a dictionary.

            Note - augmentations are applied before the adapter is called and are generally
            transforms that you only want to apply during training.
        **kwargs
            Additional keyword arguments passed to the base `PyDataset`.
        """
        super().__init__(**kwargs)

        self.dataset = dataset
        self.simulator = simulator
        self.approximator = approximator
        self.batch_size = batch_size
        self.adapter = adapter
        self.augmentations = augmentations or []

        # case offline training
        if dataset is not None:
            if num_samples:
                self.num_samples = num_samples
            else:
                self.num_samples = self._get_num_samples_from_data(dataset)
                logging.debug(f"Automatically determined {self.num_samples} samples in data.")

            self._num_batches = int(np.ceil(self.num_samples / self.batch_size))
        elif simulator is not None:
            self._num_batches = num_batches

    def __getitem__(self, index) -> dict[str, np.ndarray]:
        """
        Generate one batch of data.

        Parameters
        ----------
        item : int
            Index of the batch. Required by signature, but not used.

        Returns
        -------
        dict of str to np.ndarray
            A batch of simulated (and optionally augmented/adapted) data.
        """
        if self.simulator:
            batch = self.simulator.sample(self.batch_size)

        if self.dataset:
            index = slice(index * self.batch_size, (index + 1) * self.batch_size)
            index = np.arange(self.num_samples, dtype="int64")[index]

            batch = {
                key: np.take(value, index, axis=0) if isinstance(value, np.ndarray) else value
                for key, value in self.dataset.items()
            }

        if self.augmentations is None:
            pass
        elif isinstance(self.augmentations, Mapping):
            for key, fn in self.augmentations.items():
                batch[key] = fn(batch[key])
        elif isinstance(self.augmentations, Sequence):
            for fn in self.augmentations:
                batch = fn(batch)
        elif isinstance(self.augmentations, Callable):
            batch = self.augmentations(batch)
        else:
            raise RuntimeError(f"Could not apply augmentations of type {type(self.augmentations)}.")

        if self.adapter is not None:
            batch = self.adapter(batch)

        output = {
            "summary_inputs": summary_inputs_by_network(self.approximator, dict(batch)),
            "inference_variables": inference_variables_by_network(self.approximator, dict(batch)),
            "inference_conditions": inference_conditions_by_network(self.approximator, dict(batch)),
        }

        for k, v in output.items():
            for network_idx, tensor in output[k].items():
                output[k][network_idx] = keras.ops.convert_to_numpy(keras.ops.stop_gradient(tensor))

        return output

    def __len__(self):
        return self._num_batches

    @property
    def num_batches(self):
        return self._num_batches

    @staticmethod
    def _get_num_samples_from_data(data: Mapping) -> int:
        for key, value in data.items():
            if hasattr(value, "shape"):
                ndim = len(value.shape)
                if ndim > 1:
                    return value.shape[0]

        raise ValueError("Could not determine number of samples from data. Please pass it manually.")
