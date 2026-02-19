from collections.abc import Mapping, Sequence, Callable

import os

import numpy as np

import keras

from bayesflow.datasets import OnlineDataset, OfflineDataset, DiskDataset
from bayesflow.datasets.ensemble_dataset import EnsembleDataset
from bayesflow.networks import InferenceNetwork, PointInferenceNetwork, SummaryNetwork
from bayesflow.simulators import Simulator
from bayesflow.adapters import Adapter
from bayesflow.approximators import EnsembleApproximator, ContinuousApproximator, PointApproximator
from bayesflow.utils import find_inference_network, find_summary_network, logging, filter_kwargs

from .basic_workflow import BasicWorkflow


class EnsembleWorkflow(BasicWorkflow):
    def __init__(
        self,
        simulator: Simulator | None = None,
        adapter: Adapter | None = None,
        inference_networks: dict[str, InferenceNetwork] | InferenceNetwork | str = "coupling_flow",
        summary_networks: dict[str, SummaryNetwork] | SummaryNetwork | str | None = None,
        ensemble_size: int | None = None,
        share_inference_network: bool = False,
        initial_learning_rate: float = 5e-4,
        optimizer: keras.optimizers.Optimizer | type | None = None,
        checkpoint_filepath: str | None = None,
        checkpoint_name: str = "model",
        save_weights_only: bool = False,
        save_best_only: bool = False,
        inference_variables: Sequence[str] | str | None = None,
        inference_conditions: Sequence[str] | str | None = None,
        summary_variables: Sequence[str] | str | None = None,
        standardize: Sequence[str] | str | None = "inference_variables",
        **kwargs,
    ):
        _inference_networks = {}
        if isinstance(inference_networks, dict):
            if ensemble_size is not None:
                logging.warning(
                    "Ignoring argument ensemble_size={ensemble_size}, "
                    "because a dictionary was passed for `inference_networks`.",
                    ensemble_size=ensemble_size,
                )
            if share_inference_network:
                logging.warning(
                    "Ignoring argument share_inference_network={share_inference_network}, "
                    "because a dictionary was passed for `inference_networks`.",
                    share_inference_network=share_inference_network,
                )
            for k, v in inference_networks.items():
                _inference_networks[k] = find_inference_network(v, **kwargs.get("inference_kwargs", {}).get(k, {}))

        else:
            if ensemble_size and ensemble_size > 1:
                inference_network = find_inference_network(inference_networks, **kwargs.get("inference_kwargs", {}))
                for member_idx in range(ensemble_size):
                    member_key = f"{member_idx}"
                    if share_inference_network:
                        _inference_networks[member_key] = inference_network
                    else:
                        _inference_networks[member_key] = keras.models.clone_model(inference_network)
            elif isinstance(ensemble_size, int) and ensemble_size <= 1:
                raise ValueError("`ensemble_size` should be an integer greater than 1.")
            else:
                raise ValueError(
                    "Either `inference_networks` is a dictionary of `InferenceNetwork`s "
                    "or `ensemble_size` must be specified."
                )

        _summary_networks = {}
        if isinstance(summary_networks, dict):
            for k, v in summary_networks.items():
                if k not in _inference_networks.keys():
                    raise ValueError(f"A summary network was specified for {k}, but no inference network.")
                if v is not None:
                    _summary_networks[k] = find_summary_network(v, **kwargs.get("summary_kwargs", {}).get(k, {}))
        else:
            summary_network = find_summary_network(summary_networks, **kwargs.get("summary_kwargs", {}))
            for k in _inference_networks.keys():
                _summary_networks[k] = summary_network

        self.simulator = simulator

        adapter = adapter or BasicWorkflow.default_adapter(inference_variables, inference_conditions, summary_variables)

        approximators = {}
        for k, v in _inference_networks.items():
            if isinstance(v, PointInferenceNetwork):
                constructor = PointApproximator
            else:
                constructor = ContinuousApproximator

            approximators[k] = constructor(
                inference_network=v,
                summary_network=_summary_networks.get(k, None),
                adapter=adapter,
                standardize=standardize,
                **filter_kwargs(kwargs, keras.Model.__init__),
            )

        self.approximator = EnsembleApproximator(
            approximators=approximators, **filter_kwargs(kwargs, keras.Model.__init__)
        )

        self._init_optimizer(initial_learning_rate, optimizer, **kwargs.get("optimizer_kwargs", {}))
        self._init_checkpointing(checkpoint_filepath, checkpoint_name, save_weights_only, save_best_only)
        self.history = None

    def fit_offline(
        self,
        data: Mapping[str, np.ndarray],
        epochs: int = 100,
        batch_size: int = 32,
        data_reuse: float = 1.0,
        keep_optimizer: bool = False,
        validation_data: Mapping[str, np.ndarray] | int = None,
        augmentations: Mapping[str, Callable] | Callable = None,
        **kwargs,
    ) -> keras.callbacks.History:
        """
        Train the ensemble of approximators offline using a fixed dataset. This approach will be faster than online
        training, since no computation time is spent in generating new data for each batch,
        but it assumes that simulations can fit in memory.

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            A dictionary containing training data where keys represent variable
            names and values are corresponding realizations.
        epochs : int, optional
            The number of training epochs, by default 100. Consider increasing this number for free-form inference
            networks, such as FlowMatching or ConsistencyModel, which generally need more epochs than CouplingFlows.
        batch_size : int, optional
            The batch size used for training, by default 32.
        data_reuse : float, optional
            Similarity of training data for ensemble members in ``[0, 1]``, by default 1.0.
            See also :py:class`bayesflow.datasets.EnsembleDataset`.
        keep_optimizer : bool, optional
            Whether to retain the current state of the optimizer after training,
            by default False.
        validation_data : Mapping[str, np.ndarray] or int, optional
            A dictionary containing validation data. If an integer is provided,
            that number of validation samples will be generated (if supported).
            By default, no validation data is used.
        augmentations : dict of str to Callable or Callable, optional
            Dictionary of augmentation functions to apply to each corresponding key in the batch
            or a function to apply to the entire batch (possibly adding new keys).

            If you provide a dictionary of functions, each function should accept one element
            of your output batch and return the corresponding transformed element. Otherwise,
            your function should accept the entire dictionary output and return a dictionary.

            Note - augmentations are applied before the adapter is called and are generally
            transforms that you only want to apply during training.
        **kwargs : dict, optional
            Additional keyword arguments passed to the underlying `_fit` method.

        Returns
        -------
        history : keras.callbacks.History
            A history object containing training history, where keys correspond to
            logged metrics (e.g., loss values) and values are arrays tracking
            metric evolution over epochs.
        """

        dataset = OfflineDataset(data=data, batch_size=batch_size, adapter=self.adapter, augmentations=augmentations)

        dataset = EnsembleDataset(
            dataset, member_names=list(self.approximator.approximators.keys()), data_reuse=data_reuse
        )

        return self._fit(
            dataset,
            epochs,
            strategy="offline",
            keep_optimizer=keep_optimizer,
            validation_data=validation_data,
            **kwargs,
        )

    def fit_online(
        self,
        epochs: int = 100,
        num_batches_per_epoch: int = 100,
        batch_size: int = 32,
        data_reuse: float = 1.0,
        keep_optimizer: bool = False,
        validation_data: Mapping[str, np.ndarray] | int = None,
        augmentations: Mapping[str, Callable] | Callable = None,
        **kwargs,
    ) -> keras.callbacks.History:
        """
        Train the ensemble of approximators using an online data-generating process. The dataset is dynamically
        generated during training, making this approach suitable for scenarios where generating new simulations
        is computationally cheap.

        Parameters
        ----------
        epochs : int, optional
            The number of training epochs, by default 100.
        num_batches_per_epoch : int, optional
            The number of batches generated per epoch, by default 100.
        batch_size : int, optional
            The batch size used for training, by default 32.
        data_reuse : float, optional
            Similarity of training data for ensemble members in ``[0, 1]``, by default 1.0.
            See also :py:class`bayesflow.datasets.EnsembleDataset`.
        keep_optimizer : bool, optional
            Whether to retain the current state of the optimizer after training,
            by default False.
        validation_data : Mapping[str, np.ndarray] or int, optional
            A dictionary containing validation data. If an integer is provided,
            that number of validation samples will be generated (if supported).
            By default, no validation data is used.
        augmentations : dict of str to Callable or Callable, optional
            Dictionary of augmentation functions to apply to each corresponding key in the batch
            or a function to apply to the entire batch (possibly adding new keys).

            If you provide a dictionary of functions, each function should accept one element
            of your output batch and return the corresponding transformed element. Otherwise,
            your function should accept the entire dictionary output and return a dictionary.

            Note - augmentations are applied before the adapter is called and are generally
            transforms that you only want to apply during training.
        **kwargs : dict, optional
            Additional keyword arguments passed to the underlying `_fit` method.

        Returns
        -------
        history : keras.callbacks.History
            A history object containing training history, where keys correspond to
            logged metrics (e.g., loss values) and values are arrays tracking
            metric evolution over epochs.
        """

        dataset = OnlineDataset(
            simulator=self.simulator,
            batch_size=batch_size,
            num_batches=num_batches_per_epoch,
            adapter=self.adapter,
            augmentations=augmentations,
        )

        dataset = EnsembleDataset(
            dataset, member_names=list(self.approximator.approximators.keys()), data_reuse=data_reuse
        )

        return self._fit(
            dataset, epochs, strategy="online", keep_optimizer=keep_optimizer, validation_data=validation_data, **kwargs
        )

    def fit_disk(
        self,
        root: os.PathLike,
        pattern: str = "*.pkl",
        batch_size: int = 32,
        data_reuse: float = 1.0,
        load_fn: callable = None,
        epochs: int = 100,
        keep_optimizer: bool = False,
        validation_data: Mapping[str, np.ndarray] | int = None,
        augmentations: Mapping[str, Callable] | Callable = None,
        **kwargs,
    ) -> keras.callbacks.History:
        """
        Train the ensemble of approximators using data stored on disk. This approach is suitable for large sets of
        simulations that don't fit in memory.

        Parameters
        ----------
        root : os.PathLike
            The root directory containing the dataset files.
        pattern : str, optional
            A filename pattern to match dataset files, by default ``"*.pkl"``.
        batch_size : int, optional
            The batch size used for training, by default 32.
        data_reuse : float, optional
            Similarity of training data for ensemble members in ``[0, 1]``, by default 1.0.
            See also :py:class`bayesflow.datasets.EnsembleDataset`.
        load_fn : callable, optional
            A function to load dataset files. If None, a default loading
            function is used.
        epochs : int, optional
            The number of training epochs, by default 100. Consider increasing this number for free-form inference
            networks, such as FlowMatching or ConsistencyModel, which generally need more epochs than CouplingFlows.
        keep_optimizer : bool, optional
            Whether to retain the current state of the optimizer after training,
            by default False.
        validation_data : Mapping[str, np.ndarray] or int, optional
            A dictionary containing validation data. If an integer is provided,
            that number of validation samples will be generated (if supported).
            By default, no validation data is used.
        augmentations : dict of str to Callable or Callable, optional
            Dictionary of augmentation functions to apply to each corresponding key in the batch
            or a function to apply to the entire batch (possibly adding new keys).

            If you provide a dictionary of functions, each function should accept one element
            of your output batch and return the corresponding transformed element. Otherwise,
            your function should accept the entire dictionary output and return a dictionary.

            Note - augmentations are applied before the adapter is called and are generally
            transforms that you only want to apply during training.
        **kwargs : dict, optional
            Additional keyword arguments passed to the underlying `_fit` method.

        Returns
        -------
        history : keras.callbacks.History
            A history object containing training history, where keys correspond to
            logged metrics (e.g., loss values) and values are arrays tracking
            metric evolution over epochs.
        """

        dataset = DiskDataset(
            root=root,
            pattern=pattern,
            batch_size=batch_size,
            load_fn=load_fn,
            adapter=self.adapter,
            augmentations=augmentations,
        )

        dataset = EnsembleDataset(
            dataset, member_names=list(self.approximator.approximators.keys()), data_reuse=data_reuse
        )

        return self._fit(
            dataset,
            epochs,
            strategy="offline",
            keep_optimizer=keep_optimizer,
            validation_data=validation_data,
            **kwargs,
        )
