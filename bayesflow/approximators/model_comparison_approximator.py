from collections.abc import Mapping, Sequence

import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.datasets import OnlineDataset
from bayesflow.networks import SummaryNetwork
from bayesflow.simulators import ModelComparisonSimulator, Simulator
from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs, logging
from bayesflow.utils.serialization import serialize, deserialize, serializable

from .approximator import Approximator
from ..networks.standardization import Standardization


@serializable("bayesflow.approximators")
class ModelComparisonApproximator(Approximator):
    """
    Defines an approximator for model (simulator) comparison, where the (discrete) posterior model probabilities are
    learned with a classifier.

    Parameters
    ----------
    adapter: bf.adapters.Adapter
        Adapter for data pre-processing.
    num_models: int
        Number of models (simulators) that the approximator will compare
    classifier_network: keras.Layer
        The network backbone (e.g, an MLP) that is used for model classification.
        The input of the classifier network is created by concatenating `classifier_variables`
        and (optional) output of the summary_network.
    summary_network: bf.networks.SummaryNetwork, optional
        The summary network used for data summarization (default is None).
        The input of the summary network is `summary_variables`.
    """

    CONDITION_KEYS = ["summary_variables", "classifier_conditions"]

    def __init__(
        self,
        *,
        num_models: int,
        classifier_network: keras.Layer,
        adapter: Adapter,
        summary_network: SummaryNetwork = None,
        standardize: str | Sequence[str] | None = "all",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classifier_network = classifier_network
        self.adapter = adapter
        self.summary_network = summary_network
        self.num_models = num_models
        self.logits_projector = keras.layers.Dense(num_models)

        if isinstance(standardize, str) and standardize != "all":
            self.standardize = [standardize]
        else:
            self.standardize = standardize

        if self.standardize == "all":
            # we have to lazily initialize these
            self.standardize_layers = None
        else:
            self.standardize_layers = {var: Standardization() for var in self.standardize}

    def build_from_data(self, adapted_data: dict[str, any]):
        if self.standardize == "all":
            self.standardize = [var for var in ["summary_variables", "classifier_conditions"] if var in adapted_data]
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}
        super().build_from_data(adapted_data)

    @classmethod
    def build_adapter(
        cls,
        num_models: int,
        classifier_conditions: Sequence[str] = None,
        summary_variables: Sequence[str] = None,
        model_index_name: str = "model_indices",
    ):
        if classifier_conditions is None and summary_variables is None:
            raise ValueError("At least one of `classifier_variables` or `summary_variables` must be provided.")

        adapter = Adapter().to_array().convert_dtype("float64", "float32")

        if classifier_conditions is not None:
            adapter = adapter.concatenate(classifier_conditions, into="classifier_conditions")

        if summary_variables is not None:
            adapter = adapter.as_set(summary_variables).concatenate(summary_variables, into="summary_variables")

        adapter = (
            adapter.rename(model_index_name, "model_indices")
            .keep(["classifier_conditions", "summary_variables", "model_indices"])
            .one_hot("model_indices", num_models)
        )

        return adapter

    @classmethod
    def build_dataset(
        cls,
        *,
        dataset: keras.utils.PyDataset = None,
        simulator: ModelComparisonSimulator = None,
        simulators: Sequence[Simulator] = None,
        **kwargs,
    ) -> OnlineDataset:
        if sum(arg is not None for arg in (dataset, simulator, simulators)) != 1:
            raise ValueError("Exactly one of dataset, simulator, or simulators must be provided.")

        if simulators is not None:
            simulator = ModelComparisonSimulator(simulators)

        return super().build_dataset(dataset=dataset, simulator=simulator, **kwargs)

    def compile(
        self,
        *args,
        classifier_metrics: Sequence[keras.Metric] = None,
        summary_metrics: Sequence[keras.Metric] = None,
        **kwargs,
    ):
        if classifier_metrics:
            self.classifier_network._metrics = classifier_metrics

        if summary_metrics:
            if self.summary_network is None:
                logging.warning("Ignoring summary metrics because there is no summary network.")
            else:
                self.summary_network._metrics = summary_metrics

        return super().compile(*args, **kwargs)

    def compile_from_config(self, config):
        self.compile(**deserialize(config))
        if hasattr(self, "optimizer") and self.built:
            # Create optimizer variables.
            self.optimizer.build(self.trainable_variables)

    def compute_metrics(
        self,
        *,
        classifier_conditions: Tensor = None,
        model_indices: Tensor,
        summary_variables: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:
        """
        Computes loss and tracks metrics for the classifier and summary networks.

        This method coordinates summary metric computation (if present), combines summary outputs with
        classifier conditions, computes classifier logits and cross-entropy loss, and aggregates all
        tracked metrics into a single dictionary. Keys are prefixed with "classifier_" or "summary_"
        to indicate their origin.

        Parameters
        ----------
        classifier_conditions : Tensor, optional
            Conditioning variables for the classifier network (default is None). May be
            combined with summary network outputs if present.
        model_indices : Tensor
            Ground-truth indices or one-hot encoded labels for classification.
        summary_variables : Tensor, optional
            Input tensor(s) for the summary network (default is None). Required if a summary
            network is present.
        stage : str, optional
            Current training stage (e.g., "training", "validation", "inference"). Controls
            certain metric computations (default is "training").

        Returns
        -------
        metrics : dict[str, Tensor]
            Dictionary containing the total loss under the key "loss", as well as all tracked
            metrics for the classifier and summary networks. Each metric key is prefixed to
            indicate its source.
        """

        summary_metrics, summary_outputs = self._compute_summary_metrics(summary_variables, stage=stage)

        classifier_conditions = self._combine_conditions(classifier_conditions, summary_outputs, stage=stage)

        logits = self._compute_logits(classifier_conditions)
        cross_entropy = keras.ops.mean(keras.losses.categorical_crossentropy(model_indices, logits, from_logits=True))

        classifier_metrics = {"loss": cross_entropy}

        if stage != "training" and any(self.classifier_network.metrics):
            predictions = keras.ops.argmax(logits, axis=-1)
            classifier_metrics |= {
                metric.name: metric(model_indices, predictions) for metric in self.classifier_network.metrics
            }

        loss = classifier_metrics.get("loss") + summary_metrics.get("loss", keras.ops.zeros(()))

        classifier_metrics = {f"{key}/classifier_{key}": value for key, value in classifier_metrics.items()}
        summary_metrics = {f"{key}/summary_{key}": value for key, value in summary_metrics.items()}

        metrics = {"loss": loss} | classifier_metrics | summary_metrics
        return metrics

    def _compute_summary_metrics(self, summary_variables: Tensor, stage: str) -> tuple[dict, Tensor | None]:
        """Helper function to compute summary metrics and outputs."""
        if self.summary_network is None:
            return {}, None
        if summary_variables is None:
            raise ValueError("Summary variables are required when a summary network is present.")

        if "summary_variables" in self.standardize:
            summary_variables = self.standardize_layers["summary_variables"](summary_variables, stage=stage)

        summary_metrics = self.summary_network.compute_metrics(summary_variables, stage=stage)
        summary_outputs = summary_metrics.pop("outputs")
        return summary_metrics, summary_outputs

    def _combine_conditions(
        self, classifier_conditions: Tensor | None, summary_outputs: Tensor | None, stage
    ) -> Tensor:
        """Helper to combine classifier conditions and summary outputs, if present."""
        if classifier_conditions is None:
            return summary_outputs

        if "classifier_conditions" in self.standardize:
            classifier_conditions = self.standardize_layers["inference_conditions"](classifier_conditions, stage=stage)

        if summary_outputs is None:
            return classifier_conditions

        return keras.ops.concatenate([classifier_conditions, summary_outputs], axis=-1)

    def _compute_logits(self, classifier_conditions: Tensor) -> Tensor:
        """Helper to compute projected logits from the classifier network."""
        logits = self.classifier_network(classifier_conditions)
        logits = self.logits_projector(logits)
        return logits

    def fit(
        self,
        *,
        adapter: Adapter | str = "auto",
        dataset: keras.utils.PyDataset = None,
        simulator: ModelComparisonSimulator = None,
        simulators: Sequence[Simulator] = None,
        **kwargs,
    ):
        """
        Trains the approximator on the provided dataset or on-demand generated from the given (multi-model) simulator.
        If `dataset` is not provided, a dataset is built from the `simulator`.
        If `simulator` is not provided, it will be built from a list of `simulators`.
        If the model has not been built, it will be built using a batch from the dataset.

        Parameters
        ----------
        adapter : Adapter or str, optional
            The data adapter that will make the simulated / real outputs neural-network friendly.
        dataset : keras.utils.PyDataset, optional
            A dataset containing simulations for training. If provided, `simulator` must be None.
        simulator : ModelComparisonSimulator, optional
            A simulator used to generate a dataset. If provided, `dataset` must be None.
        simulators: Sequence[Simulator], optional
            A list of simulators (one simulator per model). If provided, `dataset` must be None.
        **kwargs : dict
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
            If both `dataset` and `simulator` or `simulators` are provided or neither is provided.
        """
        if dataset is not None:
            if simulator is not None or simulators is not None:
                raise ValueError(
                    "Received conflicting arguments. Please provide either a dataset or a simulator, but not both."
                )

            return super().fit(dataset=dataset, **kwargs)

        if adapter == "auto":
            logging.info("Building automatic data adapter.")
            adapter = self.build_adapter(num_models=self.num_models, **filter_kwargs(kwargs, self.build_adapter))

        if simulator is not None:
            return super().fit(simulator=simulator, adapter=adapter, **kwargs)

        logging.info(f"Building model comparison simulator from {len(simulators)} simulators.")

        simulator = ModelComparisonSimulator(simulators=simulators)

        return super().fit(simulator=simulator, adapter=adapter, **kwargs)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()

        config = {
            "num_models": self.num_models,
            "adapter": self.adapter,
            "classifier_network": self.classifier_network,
            "summary_network": self.summary_network,
        }

        return base_config | serialize(config)

    def get_compile_config(self):
        base_config = super().get_compile_config() or {}

        config = {
            "classifier_metrics": self.classifier_network._metrics,
            "summary_metrics": self.summary_network._metrics if self.summary_network is not None else None,
        }

        return base_config | serialize(config)

    def predict(
        self,
        *,
        conditions: Mapping[str, np.ndarray],
        logits: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """
        Predicts posterior model probabilities given input conditions. The `conditions` dictionary is preprocessed
        using the `adapter`. The output is converted to NumPy array after inference.

        Parameters
        ----------
        conditions : Mapping[str, np.ndarray]
            Dictionary of conditioning variables as NumPy arrays.
        logits: bool, default=False
            Should the posterior model probabilities be on the (unconstrained) logit space?
            If `False`, the output is a unit simplex instead.
        **kwargs : dict
            Additional keyword arguments for the adapter and classification process.

        Returns
        -------
        np.ndarray
            Predicted posterior model probabilities given `conditions`.
        """

        # Apply adapter transforms to raw simulated / real quantities
        conditions = self.adapter(conditions, strict=False, stage="inference", **kwargs)

        # Ensure only keys relevant for sampling are present in the conditions dictionary
        conditions = {k: v for k, v in conditions.items() if k in ModelComparisonApproximator.CONDITION_KEYS}
        conditions = keras.tree.map_structure(keras.ops.convert_to_tensor, conditions)

        # Optionally standardize conditions
        for key in ModelComparisonApproximator.CONDITION_KEYS:
            if key in conditions and key in self.standardize:
                conditions[key] = self.standardize_layers[key](conditions[key])

        output = self._predict(**conditions, **kwargs)

        if not logits:
            output = keras.ops.softmax(output)

        output = keras.ops.convert_to_numpy(output)

        return output

    def _predict(self, classifier_conditions: Tensor = None, summary_variables: Tensor = None, **kwargs) -> Tensor:
        if self.summary_network is None:
            if summary_variables is not None:
                raise ValueError("Cannot use summary variables without a summary network.")
        else:
            if summary_variables is None:
                raise ValueError("Summary variables are required when a summary network is present")

            summary_outputs = self.summary_network(
                summary_variables, **filter_kwargs(kwargs, self.summary_network.call)
            )

            if classifier_conditions is None:
                classifier_conditions = summary_outputs
            else:
                classifier_conditions = keras.ops.concatenate([classifier_conditions, summary_outputs], axis=1)

        output = self.classifier_network(classifier_conditions)
        output = self.logits_projector(output)

        return output

    def summaries(self, data: Mapping[str, np.ndarray], **kwargs) -> np.ndarray:
        """
        Computes the learned summary statistics of given summary variables.

        The `data` dictionary is preprocessed using the `adapter` and passed through the summary network.

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            Dictionary of simulated or real quantities as NumPy arrays.
        **kwargs : dict
            Additional keyword arguments for the adapter and the summary network.

        Returns
        -------
        summaries : np.ndarray
            The learned summary statistics.
        """
        if self.summary_network is None:
            raise ValueError("A summary network is required to compute summaries.")

        data_adapted = self.adapter(data, strict=False, stage="inference", **kwargs)
        if "summary_variables" not in data_adapted or data_adapted["summary_variables"] is None:
            raise ValueError("Summary variables are required to compute summaries.")

        summary_variables = keras.tree.map_structure(keras.ops.convert_to_tensor, data_adapted["summary_variables"])
        summaries = self.summary_network(summary_variables, **filter_kwargs(kwargs, self.summary_network.call))
        summaries = keras.ops.convert_to_numpy(summaries)

        return summaries
