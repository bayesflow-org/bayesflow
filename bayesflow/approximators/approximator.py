import multiprocessing as mp

import keras

from bayesflow.adapters import Adapter
from bayesflow.datasets import OnlineDataset
from bayesflow.simulators import Simulator
from bayesflow.utils import find_batch_size, filter_kwargs, logging

from .backend_approximators import BackendApproximator


class Approximator(BackendApproximator):
    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
        raise NotImplementedError

    @classmethod
    def build_adapter(cls, **kwargs) -> Adapter:
        # implemented by each respective architecture
        raise NotImplementedError

    def build_from_data(self, adapted_data: dict[str, any]) -> None:
        raise NotImplementedError

    @classmethod
    def build_dataset(
        cls,
        *,
        batch_size: int = "auto",
        num_batches: int,
        adapter: Adapter = "auto",
        memory_budget: str | int = "auto",
        simulator: Simulator,
        workers: int = "auto",
        use_multiprocessing: bool = False,
        max_queue_size: int = 32,
        **kwargs,
    ) -> OnlineDataset:
        if batch_size == "auto":
            batch_size = find_batch_size(memory_budget=memory_budget, sample=simulator.sample((1,)))
            logging.info(f"Using a batch size of {batch_size}.")

        if adapter == "auto":
            adapter = cls.build_adapter(**filter_kwargs(kwargs, cls.build_adapter))

        if workers == "auto":
            workers = mp.cpu_count()
            logging.info(f"Using {workers} data loading workers.")

        workers = workers or 1

        return OnlineDataset(
            simulator=simulator,
            batch_size=batch_size,
            num_batches=num_batches,
            adapter=adapter,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size,
        )

    def call(self, *args, **kwargs):
        return self.compute_metrics(*args, **kwargs)

    def fit(self, *, dataset: keras.utils.PyDataset = None, simulator: Simulator = None, **kwargs):
        """
        Trains the approximator on the provided dataset or on-demand data generated from the given simulator.
        If `dataset` is not provided, a dataset is built from the `simulator`.
        If the model has not been built, it will be built using a batch from the dataset.

        Parameters
        ----------
        dataset : keras.utils.PyDataset, optional
            A dataset containing simulations for training. If provided, `simulator` must be None.
        simulator : Simulator, optional
            A simulator used to generate a dataset. If provided, `dataset` must be None.
        **kwargs
            Additional keyword arguments passed to `keras.Model.fit()`, including (see also `build_dataset`):

            batch_size : int or None, default='auto'
                Number of samples per gradient update. Do not specify if `dataset` is provided as a
                `keras.utils.PyDataset`, `tf.data.Dataset`, `torch.utils.data.DataLoader`, or a generator function.
            epochs : int, default=1
                Number of epochs to train the model.
            verbose : {"auto", 0, 1, 2}, default="auto"
                Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks : list of keras.callbacks.Callback, optional
                List of callbacks to apply during training.
            validation_split : float, optional
                Fraction of training data to use for validation (only supported if `dataset` consists of NumPy arrays
                or tensors).
            validation_data : tuple or dataset, optional
                Data for validation, overriding `validation_split`.
            shuffle : bool, default=True
                Whether to shuffle the training data before each epoch (ignored for dataset generators).
            initial_epoch : int, default=0
                Epoch at which to start training (useful for resuming training).
            steps_per_epoch : int or None, optional
                Number of steps (batches) before declaring an epoch finished.
            validation_steps : int or None, optional
                Number of validation steps per validation epoch.
            validation_batch_size : int or None, optional
                Number of samples per validation batch (defaults to `batch_size`).
            validation_freq : int, default=1
                Specifies how many training epochs to run before performing validation.

        Returns
        -------
        keras.callbacks.History
            A history object containing the training loss and metrics values.

        Raises
        ------
        ValueError
            If both `dataset` and `simulator` are provided or neither is provided.
        """

        if dataset is None:
            if simulator is None:
                raise ValueError("Received no data to fit on. Please provide either a dataset or a simulator.")

            logging.info(f"Building dataset from simulator instance of {simulator.__class__.__name__}.")
            dataset = self.build_dataset(simulator=simulator, **filter_kwargs(kwargs, self.build_dataset))
        else:
            if simulator is not None:
                raise ValueError(
                    "Received conflicting arguments. Please provide either a dataset or a simulator, but not both."
                )

            logging.info(f"Fitting on dataset instance of {dataset.__class__.__name__}.")

        if not self.built:
            logging.info("Building on a test batch.")
            mock_data = dataset[0]
            mock_data = keras.tree.map_structure(keras.ops.convert_to_tensor, mock_data)
            mock_data_shapes = keras.tree.map_structure(keras.ops.shape, mock_data)
            self.build(mock_data_shapes)

        return super().fit(dataset=dataset, **kwargs)
