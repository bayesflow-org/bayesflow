from typing import Sequence
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import keras

from bayesflow.datasets import OnlineDataset, OfflineDataset, DiskDataset
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.simulators import Simulator
from bayesflow.adapters import Adapter
from bayesflow.approximators import ContinuousApproximator
from bayesflow.types import Shape
from bayesflow.utils import find_inference_network, find_summary_network
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
        optimizer: type = None,
        checkpoint_filepath: str = None,
        checkpoint_name: str = "model",
        save_weights_only: bool = False,
        inference_variables: Sequence[str] | str = "theta",
        inference_conditions: Sequence[str] | str = "x",
        summary_variables: Sequence[str] | str = None,
        standardize: Sequence[str] | str = "inference_variables",
        **kwargs,
    ):
        """
        BasicWorkflow class for Bayesian inference workflows.

        This class provides methods to set up, simulate, adapt, and fit models using
        simulation-based inference. It allows for both online and offline data workflows.

        Parameters
        ----------
        simulator : Simulator, optional
            A Simulator object to generate synthetic data for inference (default is None).
        adapter : Adapter, optional
            Adapter for data processing. If not provided, a default adapter will be used (default is None).
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
        inference_variables : Sequence[str] or str, optional
            Variables for inference as a sequence of strings or a single string (default is "theta").
            Important for automating diagnostics!
        inference_conditions : Sequence[str] or str, optional
            Variables used as conditions for inference (default is "x").
        summary_variables : Sequence[str] or str, optional
            Variables for summarizing data, if any (default is None).
        standardize : Sequence[str] or str, optional
            Variables to standardize during preprocessing (default is "inference_variables").
        **kwargs : dict, optional
            Additional arguments for configuring networks, adapters, optimizers, etc.
        """

        self.inference_network = find_inference_network(inference_network, **kwargs.get("inference_kwargs", {}))

        if summary_network is not None:
            self.summary_network = find_summary_network(summary_network, **kwargs.get("summary_kwargs", {}))
        else:
            self.summary_network = None

        self.simulator = simulator

        if adapter is None:
            self.adapter = BasicWorkflow.default_adapter(
                inference_variables, inference_conditions, summary_variables, standardize
            )
        else:
            self.adapter = adapter

        self.inference_variables = inference_variables

        self.approximator = ContinuousApproximator(
            inference_network=self.inference_network, summary_network=self.summary_network, adapter=self.adapter
        )

        self.initial_learning_rate = initial_learning_rate
        if isinstance(optimizer, type):
            self.optimizer = optimizer(initial_learning_rate, **kwargs.get("optimizer_kwargs", {}))
        else:
            self.optimizer = optimizer

        self.checkpoint_filepath = checkpoint_filepath
        self.checkpoint_name = checkpoint_name
        self.save_weights_only = save_weights_only
        self.history = None

    @staticmethod
    def samples_to_data_frame(samples: dict[str, np.ndarray]) -> pd.DataFrame:
        """Converts samples of a single data set to a pandas DataFrame."""
        return pd.DataFrame(keras.tree.map_structure(np.squeeze, samples))

    @staticmethod
    def default_adapter(
        inference_variables: Sequence[str] | str,
        inference_conditions: Sequence[str] | str,
        summary_variables: Sequence[str] | str,
        standardize: Sequence[str] | str,
    ) -> Adapter:
        adapter = (
            Adapter()
            .convert_dtype(from_dtype="float64", to_dtype="float32")
            .concatenate(inference_variables, into="inference_variables")
        )

        if inference_conditions is not None:
            adapter = adapter.concatenate(inference_conditions, into="inference_conditions")
        if summary_variables is not None:
            adapter = adapter.concatenate(summary_variables, into="summary_variables")

        if standardize is not None:
            adapter = adapter.standardize(include=standardize)

        return adapter

    def simulate(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        if self.simulator is not None:
            return self.simulator.sample(batch_shape, **kwargs)
        else:
            raise RuntimeError("No simulator provided!")

    def simulate_adapted(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        if self.simulator is not None:
            return self.adapter(self.simulator.sample(batch_shape, **kwargs))
        else:
            raise RuntimeError("No simulator provided!")

    def sample(
        self,
        *,
        num_samples: int,
        conditions: dict[str, np.ndarray],
        **kwargs,
    ) -> dict[str, np.ndarray]:
        return self.approximator.sample(num_samples=num_samples, conditions=conditions, **kwargs)

    def log_prob(self, data: dict[str, np.ndarray], **kwargs) -> np.ndarray:
        return self.approximator.log_prob(data=data)

    def plot_diagnostics(
        self,
        test_data: dict[str, np.ndarray] | int,
        num_samples: int = 1000,
        variable_names: Sequence[str] = None,
        as_pandas: bool = True,
        **kwargs,
    ) -> dict[str, plt.Figure]:
        if isinstance(test_data, int) and self.simulator is not None:
            test_data = self.simulator.sample(test_data, **kwargs.pop("test_data_kwargs", {}))
        elif isinstance(test_data, int):
            raise ValueError(f"No simulator found for generating {test_data} data sets.")

        if isinstance(self.inference_variables, str):
            inference_variables = {self.inference_variables: test_data.pop(self.inference_variables)}
        else:
            inference_variables = {k: test_data.pop(k) for k in self.inference_variables}

        post_samples = self.approximator.sample(
            num_samples=num_samples, conditions=test_data, **kwargs.get("approximator_kwargs", {})
        )

        figures = dict()

        if self.history is not None:
            figures["losses"] = bf_plots.loss(self.history, **kwargs.get("loss_kwargs", {}))

        figures["recovery"] = bf_plots.recovery(
            estimates=post_samples,
            targets=inference_variables,
            variable_names=variable_names,
            **kwargs.get("recovery_kwargs", {}),
        )

        figures["calibration_ecdf"] = bf_plots.calibration_ecdf(
            estimates=post_samples,
            targets=inference_variables,
            variable_names=variable_names,
            **kwargs.get("calibration_ecdf_kwargs", {}),
        )

        figures["z_score_contraction"] = bf_plots.z_score_contraction(
            estimates=post_samples,
            targets=inference_variables,
            variable_names=variable_names,
            **kwargs.get("z_score_contraction_kwargs", {}),
        )

        return figures

    def compute_diagnostics(
        self,
        test_data: dict[str, np.ndarray] | int,
        num_samples: int = 1000,
        variable_names: Sequence[str] = None,
        as_data_frame: bool = True,
        **kwargs,
    ) -> Sequence[dict] | pd.DataFrame:
        if isinstance(test_data, int) and self.simulator is not None:
            test_data = self.simulator.sample(test_data, **kwargs.pop("test_data_kwargs", {}))
        elif isinstance(test_data, int):
            raise ValueError(f"No simulator found for generating {test_data} data sets.")

        if isinstance(self.inference_variables, str):
            inference_variables = {self.inference_variables: test_data.pop(self.inference_variables)}
        else:
            inference_variables = {k: test_data.pop(k) for k in self.inference_variables}

        post_samples = self.approximator.sample(
            num_samples=num_samples, conditions=test_data, **kwargs.get("approximator_kwargs", {})
        )

        root_mean_squared_error = bf_metrics.root_mean_squared_error(
            estimates=post_samples,
            targets=inference_variables,
            variable_names=variable_names,
            **kwargs.get("root_mean_squared_error_kwargs", {}),
        )

        contraction = bf_metrics.posterior_contraction(
            estimates=post_samples,
            targets=inference_variables,
            variable_names=variable_names,
            **kwargs.get("posterior_contraction_kwargs", {}),
        )

        calibration_errors = bf_metrics.calibration_error(
            estimates=post_samples,
            targets=inference_variables,
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
                index=root_mean_squared_error["variable_names"],
            ).T
        else:
            metrics = (root_mean_squared_error, contraction, calibration_errors)

        return metrics

    def fit_disk(
        self,
        root: os.PathLike,
        pattern: str = "*.pkl",
        batch_size: int = 32,
        load_fn: callable = None,
        epochs: int = 100,
        keep_optimizer: bool = False,
        validation_data: dict[str, np.ndarray] | int = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        dataset = DiskDataset(root=root, pattern=pattern, batch_size=batch_size, load_fn=load_fn, adapter=self.adapter)

        return self._fit(
            dataset, epochs, strategy="online", keep_optimizer=keep_optimizer, validation_data=validation_data, **kwargs
        )

    def fit_online(
        self,
        epochs: int = 100,
        num_batches_per_epoch: int = 100,
        batch_size: int = 32,
        keep_optimizer: bool = False,
        validation_data: dict[str, np.ndarray] | int = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        dataset = OnlineDataset(
            simulator=self.simulator, batch_size=batch_size, num_batches=num_batches_per_epoch, adapter=self.adapter
        )

        return self._fit(
            dataset, epochs, strategy="online", keep_optimizer=keep_optimizer, validation_data=validation_data, **kwargs
        )

    def fit_offline(
        self,
        data: dict[str, np.ndarray],
        epochs: int = 100,
        batch_size: int = 32,
        keep_optimizer: bool = False,
        validation_data: dict[str, np.ndarray] | int = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        dataset = OfflineDataset(data=data, batch_size=batch_size, adapter=self.adapter)

        return self._fit(
            dataset, epochs, strategy="online", keep_optimizer=keep_optimizer, validation_data=validation_data, **kwargs
        )

    def _fit(
        self,
        dataset: keras.utils.PyDataset,
        epochs: int,
        strategy: str,
        keep_optimizer: bool,
        validation_data: dict[str, np.ndarray] | int,
        **kwargs,
    ) -> dict[str, np.ndarray]:
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
                save_best_only=True,
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
            return self.history
        except Exception as err:
            raise err
        finally:
            if not keep_optimizer:
                self.optimizer = None

    def build_optimizer(self, epochs: int, num_batches: int, strategy: str) -> keras.Optimizer | None:
        if self.optimizer is not None:
            return

        # Default case
        learning_rate = keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=epochs * num_batches,
            alpha=self.initial_learning_rate**2,
        )

        # Use adam for online learning, apply weight decay otherwise
        if strategy.lower() == "online":
            self.optimizer = keras.optimizers.Adam(learning_rate, clipnorm=1.5)
        else:
            self.optimizer = keras.optimizers.AdamW(learning_rate, weight_decay=1e-3, clipnorm=1.5)
