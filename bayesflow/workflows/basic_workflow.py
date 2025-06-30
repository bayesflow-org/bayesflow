from collections.abc import Mapping, Sequence, Callable

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import keras

from bayesflow.datasets import OnlineDataset, OfflineDataset, DiskDataset
from bayesflow.networks import InferenceNetwork, PointInferenceNetwork, SummaryNetwork
from bayesflow.simulators import Simulator
from bayesflow.adapters import Adapter
from bayesflow.approximators import ContinuousApproximator, PointApproximator
from bayesflow.types import Shape
from bayesflow.utils import find_inference_network, find_summary_network, logging
from bayesflow.diagnostics import metrics as bf_metrics
from bayesflow.diagnostics import plots as bf_plots

from .workflow import Workflow


class BasicWorkflow(Workflow):
    def __init__(
        self,
        simulator: Simulator = None,
        adapter: Adapter = None,
        inference_network: InferenceNetwork | str = "coupling_flow",
        summary_network: SummaryNetwork | str = None,
        initial_learning_rate: float = 5e-4,
        optimizer: keras.optimizers.Optimizer | type = None,
        checkpoint_filepath: str = None,
        checkpoint_name: str = "model",
        save_weights_only: bool = False,
        save_best_only: bool = False,
        inference_variables: Sequence[str] | str = None,
        inference_conditions: Sequence[str] | str = None,
        summary_variables: Sequence[str] | str = None,
        standardize: Sequence[str] | str | None = "inference_variables",
        **kwargs,
    ):
        """
        This class provides methods to set up, simulate, and fit and validate models using
        amortized Bayesian inference. It allows for both online and offline amortized workflows.

        Parameters
        ----------
        simulator : Simulator, optional
            A Simulator object to generate synthetic data for inference (default is None).
        adapter : Adapter, optional
            Adapter for data processing. If not provided, a default adapter will be used (default is None), but
            you need to make sure to provide the correct names for `inference_variables` and/or `inference_conditions`
            and/or `summary_variables`.
        inference_network : InferenceNetwork or str, optional
            The inference network used for posterior approximation, specified as an instance or by
            name (default is "coupling_flow").
        summary_network : SummaryNetwork or str, optional
            The summary network used for data summarization, specified as an instance or by name (default is None).
        initial_learning_rate : float, optional
            Initial learning rate for the optimizer (default is 5e-4).
        optimizer : type, optional
            The optimizer to be used for training. If None, a default Adam optimizer will be selected (default is None).
        checkpoint_filepath : str, optional
            Directory path where model checkpoints will be saved (default is None).
        checkpoint_name : str, optional
            Name of the checkpoint file (default is "model").
        save_weights_only : bool, optional
            If True, only the model weights will be saved during checkpointing (default is False).
        save_best_only: bool, optional
            If only the best model according to the quantity monitored (loss or validation) at the end of
            each epoch will be saved instead of the last model (default is False). Use with caution,
            as some losses (e.g. flow matching) do not reliably reflect model performance, and outliers in the
            validation data can cause unwanted effects.
        inference_variables : Sequence[str] or str, optional
            Variables for inference as a sequence of strings or a single string (default is None).
            Important for automating diagnostics!
        inference_conditions : Sequence[str] or str, optional
            Variables used as direct conditions for inference (default is None).
        summary_variables : Sequence[str] or str, optional
            Variables to be summarized through the summary network before being used as conditions (default is None).
        standardize : Sequence[str] or str, optional
            Variables to standardize during preprocessing (default is "inference_variables"). These will be
            passed to the corresponding approximator constructor and can be either "all" or any subset of
            ["inference_variables", "summary_variables", "inference_conditions"].
        **kwargs : dict, optional
            Additional arguments for configuring networks, adapters, optimizers, etc.
        """

        self.inference_network = find_inference_network(inference_network, **kwargs.get("inference_kwargs", {}))

        if summary_network is not None:
            self.summary_network = find_summary_network(summary_network, **kwargs.get("summary_kwargs", {}))
        else:
            self.summary_network = None

        self.simulator = simulator

        adapter = adapter or BasicWorkflow.default_adapter(inference_variables, inference_conditions, summary_variables)

        if isinstance(self.inference_network, PointInferenceNetwork):
            constructor = PointApproximator
        else:
            constructor = ContinuousApproximator

        self.approximator = constructor(
            inference_network=self.inference_network,
            summary_network=self.summary_network,
            adapter=adapter,
            standardize=standardize,
        )

        self.initial_learning_rate = initial_learning_rate
        if isinstance(optimizer, type):
            self.optimizer = optimizer(initial_learning_rate, **kwargs.get("optimizer_kwargs", {}))
        else:
            self.optimizer = optimizer

        self.checkpoint_filepath = checkpoint_filepath
        self.checkpoint_name = checkpoint_name
        self.save_weights_only = save_weights_only
        self.save_best_only = save_best_only
        if self.checkpoint_filepath is not None:
            if self.save_weights_only:
                file_ext = self.checkpoint_name + ".weights.h5"
            else:
                file_ext = self.checkpoint_name + ".keras"
            checkpoint_full_filepath = os.path.join(self.checkpoint_filepath, file_ext)
            if os.path.exists(checkpoint_full_filepath):
                msg = (
                    f"Checkpoint file exists: '{checkpoint_full_filepath}'.\n"
                    "Existing checkpoints can _not_ be restored/loaded using this workflow. "
                    "Upon refitting, the checkpoints will be overwritten."
                )
                if not self.save_weights_only:
                    msg += (
                        " To load the stored approximator from the checkpoint, "
                        "use approximator = keras.saving.load_model(...)"
                    )

                logging.warning(msg)
        self.history = None

    @property
    def adapter(self):
        return self.approximator.adapter

    @staticmethod
    def samples_to_data_frame(samples: Mapping[str, np.ndarray]) -> pd.DataFrame:
        """
        Convert a dictionary of samples into a pandas DataFrame.

        Parameters
        ----------
        samples : Mapping[str, np.ndarray]
            A dictionary where keys represent variable names and values are
            arrays containing sampled data.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame where each column corresponds to a variable,
            and rows represent individual samples.
        """
        return pd.DataFrame(keras.tree.map_structure(np.squeeze, samples))

    @staticmethod
    def default_adapter(
        inference_variables: Sequence[str] | str,
        inference_conditions: Sequence[str] | str,
        summary_variables: Sequence[str] | str,
    ) -> Adapter:
        """
        Create a default adapter for processing inference variables, conditions,
        summaries, and standardization.

        - Converts all float64 values to float32 for computational efficiency.

        Parameters
        ----------
        inference_variables : Sequence[str] or str
            The variables to be treated as inference targets.
        inference_conditions : Sequence[str] or str
            The variables used as conditions for inference.
        summary_variables : Sequence[str] or str
            The variables used for summarization.

        Returns
        -------
        Adapter
            A configured Adapter instance that applies dtype conversion,
            concatenation, and optional standardization.
        """

        adapter = (
            Adapter()
            .convert_dtype(from_dtype="float64", to_dtype="float32")
            .concatenate(inference_variables, into="inference_variables")
        )

        if inference_conditions is not None:
            adapter = adapter.concatenate(inference_conditions, into="inference_conditions")
        if summary_variables is not None:
            adapter = adapter.concatenate(summary_variables, into="summary_variables")

        return adapter

    def simulate(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        """
        Generates a batch of simulations using the provided simulator.

        Parameters
        ----------
        batch_shape : Shape
            The shape of the batch to be simulated. Typically an integer for simple simulators.
        **kwargs : dict, optional
            Additional keyword arguments passed to the simulator's sample method.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary where keys represent variable names and values are
            NumPy arrays containing the simulated variables.

        Raises
        ------
        RuntimeError
            If no simulator is provided.
        """
        if self.simulator is not None:
            return self.simulator.sample(batch_shape, **kwargs)
        raise RuntimeError("No simulator provided!")

    def simulate_adapted(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        """
        Generates a batch of simulations and applies the adapter to the result.

        Parameters
        ----------
        batch_shape : Shape
            The shape of the batch to be simulated. Typically an integer for simple simulators.
        **kwargs : dict, optional
            Additional keyword arguments passed to the simulator's sample method.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary where keys represent variable names and values are
            NumPy arrays containing the adapted simulated variables.

        Raises
        ------
        RuntimeError
            If no simulator is provided.
        """
        if self.simulator is not None:
            return self.adapter(self.simulator.sample(batch_shape, **kwargs))
        raise RuntimeError("No simulator provided!")

    def sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Draws `num_samples` samples from the approximator given specified conditions.

        Parameters
        ----------
        num_samples : int
            The number of samples to generate.
        conditions : dict[str, np.ndarray]
            A dictionary where keys represent variable names and values are
            NumPy arrays containing the adapted simulated variables. Keys used as summary or inference
            conditions during training should be present.
        **kwargs : dict, optional
            Additional keyword arguments passed to the approximator's sampling function.

        Returns
        -------
        dict[str, np.ndarray]
            A dictionary where keys correspond to variable names and
            values are arrays containing the generated samples.
        """
        return self.approximator.sample(num_samples=num_samples, conditions=conditions, **kwargs)

    def estimate(
        self,
        *,
        conditions: Mapping[str, np.ndarray],
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray | dict[str, np.ndarray]]]:
        """
        Estimates point summaries of inference variables based on specified conditions.

        Parameters
        ----------
        conditions : Mapping[str, np.ndarray]
            A dictionary mapping variable names to arrays representing the conditions for the estimation process.
        **kwargs
            Additional keyword arguments passed to underlying processing functions.

        Returns
        -------
        estimates : dict[str, dict[str, np.ndarray or dict[str, np.ndarray]]]
            The estimates of inference variables in a nested dictionary.

            1. Each first-level key is the name of an inference variable.
            2. Each second-level key is the name of a scoring rule.
            3. (If the scoring rule comprises multiple estimators, each third-level key is the name of an estimator.)

            Each estimator output (i.e., dictionary value that is not itself a dictionary) is an array
            of shape (num_datasets, point_estimate_size, variable_block_size).
        """
        return self.approximator.estimate(conditions=conditions, **kwargs)

    def log_prob(self, data: Mapping[str, np.ndarray], **kwargs) -> np.ndarray:
        """
        Compute the log probability of given variables under the approximator.

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            A dictionary where keys represent variable names and values are arrays corresponding to the variables'
            realizations.
        **kwargs : dict, optional
            Additional keyword arguments passed to the approximator's log probability function.

        Returns
        -------
        np.ndarray
            An array containing the log probabilities computed from the provided variables.
        """
        return self.approximator.log_prob(data=data, **kwargs)

    def plot_default_diagnostics(
        self,
        test_data: Mapping[str, np.ndarray] | int,
        num_samples: int = 1000,
        variable_keys: Sequence[str] = None,
        variable_names: Sequence[str] = None,
        **kwargs,
    ) -> dict[str, plt.Figure]:
        """
        Generates default diagnostic plots to evaluate the quality of inference. The function produces several
        diagnostic plots, including
        - Loss history (if training history is available).
        - Parameter recovery plots.
        - Calibration ECDF plots.
        - Z-score contraction plots.

        Parameters
        ----------
        test_data : Mapping[str, np.ndarray] or int
            A dictionary containing test data where keys represent variable
            names and values are corresponding data arrays. If an integer
            is provided, that number of test data sets will be generated
            using the simulator (if available).
        num_samples : int, optional
            The number of samples to draw from the approximator for diagnostics,
            by default 1000.
        variable_keys : list or None, optional, default: None
           Select keys from the dictionaries provided in estimates and targets.
           By default, select all keys.
        variable_names : list or None, optional, default: None
            The variable names for nice table plot titles.
        **kwargs : dict, optional
            Additional keyword arguments:

            - `test_data_kwargs`: dict, optional
                Arguments to pass to the simulator when generating test data.
            - `approximator_kwargs`: dict, optional
                Arguments to pass to the approximator's sampling function.
            - `loss_kwargs`: dict, optional
                Arguments for customizing the loss plot.
            - `recovery_kwargs`: dict, optional
                Arguments for customizing the parameter recovery plot.
            - `calibration_ecdf_kwargs`: dict, optional
                Arguments for customizing the empirical cumulative distribution
                function (ECDF) calibration plot.
            - `z_score_contraction_kwargs`: dict, optional
                Arguments for customizing the z-score contraction plot.

        Returns
        -------
        dict[str, plt.Figure]
            A dictionary where keys correspond to different diagnostic plot
            types, and values are the respective matplotlib Figure objects.
        """

        samples, test_data = self._prepare_for_diagnostics(test_data, num_samples, **kwargs)

        figures = dict()

        if self.history is not None:
            figures["losses"] = bf_plots.loss(self.history, **kwargs.get("loss_kwargs", {}))

        plot_fns = {
            "recovery": bf_plots.recovery,
            "calibration_ecdf": bf_plots.calibration_ecdf,
            "z_score_contraction": bf_plots.z_score_contraction,
        }

        for k, plot_fn in plot_fns.items():
            figures[k] = plot_fn(
                estimates=samples,
                targets=test_data,
                variable_keys=variable_keys,
                variable_names=variable_names,
                **kwargs.get(f"{k}_kwargs", {}),
            )

        return figures

    def plot_custom_diagnostics(
        self,
        test_data: Mapping[str, np.ndarray] | int,
        plot_fns: Mapping[str, Callable],
        num_samples: int = 1000,
        variable_keys: Sequence[str] = None,
        variable_names: Sequence[str] = None,
        **kwargs,
    ) -> dict[str, plt.Figure]:
        """
        Generates custom diagnostic plots to evaluate the quality of inference. The functions passed should have
        the following signature:
        - fn(samples, inference_variables, variable_names)

        They should also return a single matplotlib Figure object.

        Parameters
        ----------
        test_data : Mapping[str, np.ndarray] or int
            A dictionary containing test data where keys represent variable
            names and values are corresponding data arrays. If an integer
            is provided, that number of test data sets will be generated
            using the simulator (if available).
        plot_fns: Mapping[str, Callable]
            A dictionary containing custom plotting functions where keys represent
            the function names and values correspond to the functions themselves.
            The functions should have a signature of fn(samples, inference_variables, variable_names)
        num_samples : int, optional
            The number of samples to draw from the approximator for diagnostics,
            by default 1000.
        variable_keys : list or None, optional, default: None
           Select keys from the dictionaries provided in estimates and targets.
           By default, select all keys.
        variable_names : list or None, optional, default: None
            The variable names for nice table plot titles.
        **kwargs : dict, optional
            Additional keyword arguments:

            - `test_data_kwargs`: dict, optional
                Arguments to pass to the simulator when generating test data.
            - `approximator_kwargs`: dict, optional
                Arguments to pass to the approximator's sampling function.
            - `loss_kwargs`: dict, optional
                Arguments for customizing the loss plot.
            - `recovery_kwargs`: dict, optional
                Arguments for customizing the parameter recovery plot.
            - `calibration_ecdf_kwargs`: dict, optional
                Arguments for customizing the empirical cumulative distribution
                function (ECDF) calibration plot.
            - `z_score_contraction_kwargs`: dict, optional
                Arguments for customizing the z-score contraction plot.

        Returns
        -------
        dict[str, plt.Figure]
            A dictionary where keys correspond to different diagnostic plot
            types, and values are the respective matplotlib Figure objects.
        """

        samples, test_data = self._prepare_for_diagnostics(test_data, num_samples, **kwargs)

        figures = dict()
        for key, plot_fn in plot_fns.items():
            figures[key] = plot_fn(samples, test_data, variable_keys=variable_keys, variable_names=variable_names)
        return figures

    def plot_diagnostics(self, **kwargs):
        logging.warning(
            "This function will be deprecated in future versions. Please, use plot_default_diagnostics"
            "or plot_custom_diagnositcs if you want to use your custom diagnostics."
        )
        return self.plot_default_diagnostics(**kwargs)

    def compute_default_diagnostics(
        self,
        test_data: Mapping[str, np.ndarray] | int,
        num_samples: int = 1000,
        variable_keys: Sequence[str] = None,
        variable_names: Sequence[str] = None,
        as_data_frame: bool = True,
        **kwargs,
    ) -> Sequence[dict] | pd.DataFrame:
        """
        Computes default diagnostic metrics to evaluate the quality of inference. The function computes several
        diagnostic metrics, including:
        - Root Mean Squared Error (RMSE)
        - Posterior contraction
        - Calibration error

        Parameters
        ----------
        test_data : Mapping[str, np.ndarray] or int
            A dictionary containing test data where keys represent variable
            names and values are corresponding realizations. If an integer
            is provided, that number of test data sets will be generated
            using the simulator (if available).
        num_samples : int, optional
            The number of samples to draw from the approximator for diagnostics,
            by default 1000.
        variable_keys : list or None, optional, default: None
           Select keys from the dictionaries provided in estimates and targets.
           By default, select all keys.
        variable_names : list or None, optional, default: None
            The parameter names for nice table plot titles.
        as_data_frame : bool, optional
            Whether to return the results as a pandas DataFrame (default: True).
            If False, a sequence of dictionaries with metric values is returned.
        **kwargs : dict, optional
            Additional keyword arguments:

            - `test_data_kwargs`: dict, optional
                Arguments to pass to the simulator when generating test data.
            - `approximator_kwargs`: dict, optional
                Arguments to pass to the approximator's sampling function.
            - `root_mean_squared_error_kwargs`: dict, optional
                Arguments for customizing the RMSE computation.
            - `posterior_contraction_kwargs`: dict, optional
                Arguments for customizing the posterior contraction computation.
            - `calibration_error_kwargs`: dict, optional
                Arguments for customizing the calibration error computation.

        Returns
        -------
        Sequence[dict] or pd.DataFrame
            If `as_data_frame` is True, returns a pandas DataFrame containing
            the computed diagnostic metrics for each variable. Otherwise,
            returns a sequence of dictionaries with metric values.
        """

        samples, test_data = self._prepare_for_diagnostics(test_data, num_samples, **kwargs)

        root_mean_squared_error = bf_metrics.root_mean_squared_error(
            estimates=samples,
            targets=test_data,
            variable_keys=variable_keys,
            variable_names=variable_names,
            **kwargs.get("root_mean_squared_error_kwargs", {}),
        )

        contraction = bf_metrics.posterior_contraction(
            estimates=samples,
            targets=test_data,
            variable_keys=variable_keys,
            variable_names=variable_names,
            **kwargs.get("posterior_contraction_kwargs", {}),
        )

        calibration_errors = bf_metrics.calibration_error(
            estimates=samples,
            targets=test_data,
            variable_keys=variable_keys,
            variable_names=variable_names,
            **kwargs.get("calibration_error_kwargs", {}),
        )

        if as_data_frame:
            metrics = pd.DataFrame(
                {
                    root_mean_squared_error["metric_name"]: root_mean_squared_error["values"],
                    contraction["metric_name"]: contraction["values"],
                    calibration_errors["metric_name"]: calibration_errors["values"],
                },
                index=variable_keys or root_mean_squared_error["variable_names"],
            ).T
        else:
            metrics = (root_mean_squared_error, contraction, calibration_errors)

        return metrics

    def compute_custom_diagnostics(
        self,
        test_data: Mapping[str, np.ndarray] | int,
        metrics: Mapping[str, Callable],
        num_samples: int = 1000,
        variable_keys: Sequence[str] = None,
        variable_names: Sequence[str] = None,
        as_data_frame: bool = True,
        **kwargs,
    ) -> Sequence[Mapping] | pd.DataFrame:
        """
        Computes custom diagnostic metrics to evaluate the quality of inference. The metric functions should
        have a signature of:

        - metric_fn(samples, inference_variables, variable_names, variable_keys) or
        - metric_fn(samples, inference_variables, **kwargs)

        And return a dictionary containing the metric name in 'name' key and the metric values in a 'values' key.

        Parameters
        ----------
        test_data : Mapping[str, np.ndarray] or int
            A dictionary containing test data where keys represent variable
            names and values are corresponding realizations. If an integer
            is provided, that number of test data sets will be generated
            using the simulator (if available).
        metrics: Mapping[str, Callable]
            A dictionary containing custom metric functions where keys represent
            the function names and values correspond to the functions themselves.
            The functions should have a signature of fn(samples, inference_variables, variable_names)
        num_samples : int, optional
            The number of samples to draw from the approximator for diagnostics,
            by default 1000.
        variable_keys : list or None, optional, default: None
           Select keys from the dictionaries provided in estimates and targets.
           By default, select all keys.
        variable_names : list or None, optional, default: None
            The variable names for nice plot titles.
        as_data_frame : bool, optional
            Whether to return the results as a pandas DataFrame (default: True).
            If False, a sequence of dictionaries with metric values is returned.
        **kwargs : dict, optional
            Additional keyword arguments:

            - `test_data_kwargs`: dict, optional
                Arguments to pass to the simulator when generating test data.
            - `approximator_kwargs`: dict, optional
                Arguments to pass to the approximator's sampling function.
            - `root_mean_squared_error_kwargs`: dict, optional
                Arguments for customizing the RMSE computation.
            - `posterior_contraction_kwargs`: dict, optional
                Arguments for customizing the posterior contraction computation.
            - `calibration_error_kwargs`: dict, optional
                Arguments for customizing the calibration error computation.

        Returns
        -------
        Sequence[dict] or pd.DataFrame
            If `as_data_frame` is True, returns a pandas DataFrame containing
            the computed diagnostic metrics for each variable. Otherwise,
            returns a sequence of dictionaries with metric values.
        """

        samples, test_data = self._prepare_for_diagnostics(test_data, num_samples, **kwargs)

        metrics_dict = {}
        for key, metric_fn in metrics.items():
            metric = metric_fn(samples, test_data, variable_keys=variable_keys, variable_names=variable_names)
            metrics_dict[metric["name"]] = metric["values"]

        if as_data_frame:
            return pd.DataFrame(metrics_dict, index=variable_names)
        return metrics_dict

    def compute_diagnostics(self, **kwargs):
        logging.warning(
            "This function will be deprecated in future versions. Please, use plot_default_diagnostics"
            "or compute_custom_diagnositcs if you want to use your own metrics."
        )
        return self.compute_default_diagnostics(**kwargs)

    def fit_offline(
        self,
        data: Mapping[str, np.ndarray],
        epochs: int = 100,
        batch_size: int = 32,
        keep_optimizer: bool = False,
        validation_data: Mapping[str, np.ndarray] | int = None,
        augmentations: Mapping[str, Callable] | Callable = None,
        **kwargs,
    ) -> keras.callbacks.History:
        """
        Train the approximator offline using a fixed dataset. This approach will be faster than online training,
        since no computation time is spent in generating new data for each batch, but it assumes that simulations
        can fit in memory.

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

        return self._fit(
            dataset, epochs, strategy="online", keep_optimizer=keep_optimizer, validation_data=validation_data, **kwargs
        )

    def fit_online(
        self,
        epochs: int = 100,
        num_batches_per_epoch: int = 100,
        batch_size: int = 32,
        keep_optimizer: bool = False,
        validation_data: Mapping[str, np.ndarray] | int = None,
        augmentations: Mapping[str, Callable] | Callable = None,
        **kwargs,
    ) -> keras.callbacks.History:
        """
        Train the approximator using an online data-generating process. The dataset is dynamically generated during
        training, making this approach suitable for scenarios where generating new simulations is computationally cheap.

        Parameters
        ----------
        epochs : int, optional
            The number of training epochs, by default 100.
        num_batches_per_epoch : int, optional
            The number of batches generated per epoch, by default 100.
        batch_size : int, optional
            The batch size used for training, by default 32.
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

        return self._fit(
            dataset, epochs, strategy="online", keep_optimizer=keep_optimizer, validation_data=validation_data, **kwargs
        )

    def fit_disk(
        self,
        root: os.PathLike,
        pattern: str = "*.pkl",
        batch_size: int = 32,
        load_fn: callable = None,
        epochs: int = 100,
        keep_optimizer: bool = False,
        validation_data: Mapping[str, np.ndarray] | int = None,
        augmentations: Mapping[str, Callable] | Callable = None,
        **kwargs,
    ) -> keras.callbacks.History:
        """
        Train the approximator using data stored on disk. This approach is suitable for large sets of simulations that
        don't fit in memory.

        Parameters
        ----------
        root : os.PathLike
            The root directory containing the dataset files.
        pattern : str, optional
            A filename pattern to match dataset files, by default ``"*.pkl"``.
        batch_size : int, optional
            The batch size used for training, by default 32.
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

        return self._fit(
            dataset, epochs, strategy="online", keep_optimizer=keep_optimizer, validation_data=validation_data, **kwargs
        )

    def build_optimizer(self, epochs: int, num_batches: int, strategy: str) -> keras.Optimizer | None:
        """
        Build and initialize the optimizer based on the training strategy. Uses a cosine decay learning rate schedule,
        where the final learning rate is proportional to the square of the initial learning rate, as found to work
        best in SBI.

        The default optimizer will use 5% of the epochs as warmup; during the warmup phase, the learning rate
        will be increased from 10% of the initial learning rate to initial learning rate supplied to the workflow.

        Parameters
        ----------
        epochs : int
            The total number of training epochs.
        num_batches : int
            The number of batches per epoch.
        strategy : str
            The training strategy, either "online" or another mode that
            applies weight decay. For "online" training, an Adam optimizer with gradient clipping is used. For other
            strategies, AdamW is used with weight decay to encourage regularization.

        Returns
        -------
        keras.Optimizer or None
            The initialized optimizer if it was not already set. Returns None
            if the optimizer was already defined.
        """

        if self.optimizer is not None:
            return

        total_steps = int(epochs * num_batches)
        warmup_steps = int(0.05 * epochs * num_batches)
        decay_steps = total_steps - warmup_steps

        # Default case
        learning_rate = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=0.1 * self.initial_learning_rate,
            warmup_target=self.initial_learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            alpha=0,
        )

        # Use adam for online learning, apply weight decay otherwise
        if strategy.lower() == "online":
            self.optimizer = keras.optimizers.Adam(learning_rate, clipnorm=1.5)
        else:
            self.optimizer = keras.optimizers.AdamW(learning_rate, weight_decay=5e-3, clipnorm=1.5)

    def _fit(
        self,
        dataset: keras.utils.PyDataset,
        epochs: int,
        strategy: str,
        keep_optimizer: bool,
        validation_data: Mapping[str, np.ndarray] | int,
        **kwargs,
    ) -> keras.callbacks.History:
        if validation_data is not None:
            if isinstance(validation_data, int) and self.simulator is not None:
                validation_data = self.simulator.sample(validation_data, **kwargs.pop("validation_data_kwargs", {}))
            elif isinstance(validation_data, int):
                raise ValueError(f"No simulator found for generating {validation_data} data sets.")

            validation_data = OfflineDataset(data=validation_data, batch_size=dataset.batch_size, adapter=self.adapter)
            monitor = "val_loss"
        else:
            monitor = "loss"

        if self.checkpoint_filepath is not None:
            if self.save_weights_only:
                file_ext = self.checkpoint_name + ".weights.h5"
            else:
                file_ext = self.checkpoint_name + ".keras"

            model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
                filepath=os.path.join(self.checkpoint_filepath, file_ext),
                monitor=monitor,
                mode="min",
                save_best_only=self.save_best_only,
                save_weights_only=self.save_weights_only,
                save_freq="epoch",
            )

            if kwargs.get("callbacks") is not None:
                kwargs["callbacks"].append(model_checkpoint_callback)
            else:
                kwargs["callbacks"] = [model_checkpoint_callback]

        self.build_optimizer(epochs, dataset.num_batches, strategy=strategy)

        if not self.approximator.built:
            self.approximator.compile(optimizer=self.optimizer, metrics=kwargs.pop("metrics", None))

        try:
            self.history = self.approximator.fit(
                dataset=dataset, epochs=epochs, validation_data=validation_data, **kwargs
            )
            self._on_training_finished()
            return self.history
        finally:
            if not keep_optimizer:
                self.optimizer = None

    def _prepare_for_diagnostics(self, test_data: Mapping[str, np.ndarray] | int, num_samples: int = 1000, **kwargs):
        if isinstance(test_data, int) and self.simulator is not None:
            test_data = self.simulator.sample(test_data, **kwargs.pop("test_data_kwargs", {}))
        elif isinstance(test_data, int):
            raise ValueError(f"No simulator found for generating {test_data} data sets.")

        samples = self.approximator.sample(
            num_samples=num_samples, conditions=test_data, **kwargs.get("approximator_kwargs", {})
        )

        return samples, test_data

    def _on_training_finished(self):
        if self.checkpoint_filepath is not None:
            if self.save_weights_only:
                file_ext = self.checkpoint_name + ".weights.h5"
            else:
                file_ext = self.checkpoint_name + ".keras"

            logging.info(f"""Training is now finished.
            You can find the trained approximator at '{self.checkpoint_filepath}/{self.checkpoint_name}.{file_ext}'.
            To load it, use approximator = keras.saving.load_model(...).""")
