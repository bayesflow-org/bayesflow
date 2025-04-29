from collections.abc import Mapping, Sequence

import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.datasets import OnlineDataset
from bayesflow.networks import SummaryNetwork
from bayesflow.simulators import ModelComparisonSimulator, Simulator
from bayesflow.types import Shape, Tensor
from bayesflow.utils import filter_kwargs, logging
from bayesflow.utils.serialization import serialize, deserialize, serializable

from .approximator import Approximator


@serializable
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

    SAMPLE_KEYS = ["summary_variables", "classifier_conditions"]

    def __init__(
        self,
        *,
        num_models: int,
        classifier_network: keras.Layer,
        adapter: Adapter,
        summary_network: SummaryNetwork = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classifier_network = classifier_network
        self.adapter = adapter
        self.summary_network = summary_network
        self.num_models = num_models
        self.logits_projector = keras.layers.Dense(num_models)

    def build(self, data_shapes: Mapping[str, Shape]):
        data = {key: keras.ops.zeros(value) for key, value in data_shapes.items()}
        self.compute_metrics(**data, stage="training")

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
            .standardize(exclude="model_indices")
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
        if self.summary_network is None:
            summary_metrics = {}
        else:
            summary_metrics = self.summary_network.compute_metrics(summary_variables, stage=stage)
            summary_outputs = summary_metrics.pop("outputs")

            if classifier_conditions is None:
                classifier_conditions = summary_outputs
            else:
                classifier_conditions = keras.ops.concatenate([classifier_conditions, summary_outputs], axis=-1)

        # we could move this into its own class
        logits = self.classifier_network(classifier_conditions)
        logits = self.logits_projector(logits)

        cross_entropy = keras.losses.categorical_crossentropy(model_indices, logits, from_logits=True)
        cross_entropy = keras.ops.mean(cross_entropy)

        classifier_metrics = {"loss": cross_entropy}

        if stage != "training" and any(self.classifier_network.metrics):
            # compute sample-based metrics
            predictions = keras.ops.argmax(logits, axis=-1)
            classifier_metrics |= {
                metric.name: metric(model_indices, predictions) for metric in self.classifier_network.metrics
            }

        loss = classifier_metrics.get("loss", keras.ops.zeros(())) + summary_metrics.get("loss", keras.ops.zeros(()))

        classifier_metrics = {f"{key}/classifier_{key}": value for key, value in classifier_metrics.items()}
        summary_metrics = {f"{key}/summary_{key}": value for key, value in summary_metrics.items()}

        metrics = {"loss": loss} | classifier_metrics | summary_metrics

        return metrics

    def fit(
        self,
        *,
        adapter: Adapter = "auto",
        dataset: keras.utils.PyDataset = None,
        simulator: ModelComparisonSimulator = None,
        simulators: Sequence[Simulator] = None,
        **kwargs,
    ):
        """
        Trains the approximator on the provided dataset or on-demand generated from the given (multi-model) simulator.
        If `dataset` is not provided, a dataset is built from the `simulator`.
        If `simulator` is not provided, it will be build from a list of `simulators`.
        If the model has not been built, it will be built using a batch from the dataset.

        Parameters
        ----------
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
        conditions = {k: v for k, v in conditions.items() if k in ModelComparisonApproximator.SAMPLE_KEYS}

        conditions = keras.tree.map_structure(keras.ops.convert_to_tensor, conditions)

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
