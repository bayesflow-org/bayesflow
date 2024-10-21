from collections.abc import Mapping, Sequence

import keras
from keras.saving import (
    deserialize_keras_object as deserialize,
    register_keras_serializable as serializable,
    serialize_keras_object as serialize,
)

from bayesflow.data_adapters import DataAdapter
from bayesflow.datasets import OnlineDataset
from bayesflow.networks import SummaryNetwork
from bayesflow.simulators import ModelComparisonSimulator, Simulator
from bayesflow.types import Shape, Tensor
from bayesflow.utils import filter_kwargs, logging
from .approximator import Approximator


@serializable(package="bayesflow.approximators")
class ModelComparisonApproximator(Approximator):
    """
    Defines an approximator for model (simulator) comparison, where the (discrete) posterior model probabilities are
    learned with a classifier.
    """

    def __init__(
        self,
        *,
        num_models: int,
        classifier_network: keras.Layer,
        data_adapter: DataAdapter,
        summary_network: SummaryNetwork = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.classifier_network = classifier_network
        self.data_adapter = data_adapter
        self.summary_network = summary_network

        self.logits_projector = keras.layers.Dense(num_models)

    def build(self, data_shapes: Mapping[str, Shape]):
        data = {key: keras.ops.zeros(value) for key, value in data_shapes.items()}
        self.compute_metrics(**data, stage="training")

    @classmethod
    def build_data_adapter(
        cls,
        classifier_conditions: Sequence[str] = None,
        summary_variables: Sequence[str] = None,
        model_index_name: str = "model_indices",
    ):
        if classifier_conditions is None and summary_variables is None:
            raise ValueError("At least one of `classifier_variables` or `summary_variables` must be provided.")

        data_adapter = DataAdapter().to_array().convert_dtype("float64", "float32")

        if classifier_conditions is not None:
            data_adapter = data_adapter.concatenate(classifier_conditions, into="classifier_conditions")

        if summary_variables is not None:
            data_adapter = data_adapter.as_set(summary_variables).concatenate(
                summary_variables, into="summary_variables"
            )

        data_adapter = (
            data_adapter.rename(model_index_name, "model_indices")
            .keep(["classifier_conditions", "summary_variables", "model_indices"])
            .standardize(exclude="model_indices")
        )

        # TODO: add one-hot encoding
        # .one_hot("model_indices", self.num_models)

        return data_adapter

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
        data_adapter: DataAdapter = "auto",
        dataset: keras.utils.PyDataset = None,
        simulator: ModelComparisonSimulator = None,
        simulators: Sequence[Simulator] = None,
        **kwargs,
    ):
        if dataset is not None:
            if simulator is not None or simulators is not None:
                raise ValueError(
                    "Received conflicting arguments. Please provide either a dataset or a simulator, but not both."
                )

            return super().fit(dataset=dataset, **kwargs)

        if data_adapter == "auto":
            logging.info("Building automatic data adapter.")
            data_adapter = self.build_data_adapter(**filter_kwargs(kwargs, self.build_data_adapter))

        if simulator is not None:
            return super().fit(simulator=simulator, data_adapter=data_adapter, **kwargs)

        logging.info(f"Building model comparison simulator from {len(simulators)} simulators.")

        simulator = ModelComparisonSimulator(simulators=simulators)

        return super().fit(simulator=simulator, data_adapter=data_adapter, **kwargs)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        data_adapter = deserialize(config["data_adapter"], custom_objects=custom_objects)
        classifier_network = deserialize(config["classifier_network"], custom_objects=custom_objects)
        summary_network = deserialize(config["summary_network"], custom_objects=custom_objects)
        return cls(
            data_adapter=data_adapter, classifier_network=classifier_network, summary_network=summary_network, **config
        )

    def get_config(self):
        base_config = super().get_config()

        config = {
            "data_adapter": serialize(self.data_adapter),
            "classifier_network": serialize(self.classifier_network),
            "summary_network": serialize(self.summary_network),
        }

        return base_config | config
