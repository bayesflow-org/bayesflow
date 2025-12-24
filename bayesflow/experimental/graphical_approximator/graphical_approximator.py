from collections.abc import Mapping, Sequence

import keras
import numpy as np

# TODO: add log_prob method to approximator
from ...adapters import Adapter
from ...approximators import Approximator
from ...networks import InferenceNetwork, SummaryNetwork
from ...networks.standardization import Standardization
from ...types import Shape
from ..graphical_simulator import SimulationOutput
from ..graphs import InvertedGraph
from .utils import (
    inference_condition_shapes_by_network,
    inference_conditions_by_network,
    inference_variable_shapes_by_network,
    inference_variables_by_network,
    prepare_inference_conditions,
    split_network_output,
    summary_input_shapes_by_network,
    summary_inputs_by_network,
    summary_outputs_by_network,
)


class GraphicalApproximator(Approximator):
    def __init__(
        self,
        graph: InvertedGraph,
        *,
        adapter: Adapter,
        inference_networks: Sequence[InferenceNetwork],
        summary_networks: Sequence[SummaryNetwork] | None = None,
        standardize: str | Sequence[str] | None = "all",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.graph = graph
        self.adapter = adapter
        self.inference_networks = inference_networks
        self.summary_networks = summary_networks
        self.data_shapes = None

        if isinstance(standardize, str) and standardize != "all":
            self.standardize = [standardize]
        else:
            self.standardize = standardize or []

        if standardize == "all":
            self.standardize_layers = None
        else:
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

    def build(self, data_shapes: dict[str, Shape]) -> None:
        # build summary networks
        input_shapes = summary_input_shapes_by_network(self, data_shapes)
        for i, summary_network in enumerate(self.summary_networks or []):
            if not summary_network.built:
                summary_network.build(input_shapes[i])

        # build inference networks
        variable_shapes = inference_variable_shapes_by_network(self, data_shapes)
        condition_shapes = inference_condition_shapes_by_network(self, data_shapes)

        for i, inference_network in enumerate(self.inference_networks or []):
            if not inference_network.built:
                inference_network.build(variable_shapes[i], condition_shapes[i])

        # build standardization layers
        if self.standardize == "all":
            # Only include variables present in data_shapes
            self.standardize = list(data_shapes.keys())
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

        for var in self.standardize:
            self.standardize_layers[var].build(data_shapes[var])

        self.data_shapes = data_shapes
        self.built = True

    def compute_metrics(self, stage: str = "training", **kwargs):
        # compute summary metrics
        summary_inputs = summary_inputs_by_network(self, kwargs)
        summary_metrics = {}

        for i, summary_network in enumerate(self.summary_networks or []):
            summary_metrics[i] = summary_network.compute_metrics(summary_inputs[i], stage=stage)
            summary_metrics[i].pop("outputs")

        # compute inference metrics
        inference_conditions = inference_conditions_by_network(self, kwargs)
        inference_variables = inference_variables_by_network(self, kwargs)

        inference_metrics = {}
        for i, inference_network in enumerate(self.inference_networks):
            inference_metrics[i] = inference_network.compute_metrics(
                inference_variables[i], conditions=inference_conditions[i], stage=stage
            )

        # combine metrics
        total_loss = 0
        combined_metrics = {}

        for i, metric_type in enumerate([summary_metrics, inference_metrics]):
            prefix = "summary_metrics" if i == 0 else "infrence_metrics"
            for val, metrics in metric_type.items():
                if "loss" in metrics:
                    total_loss += metrics["loss"]
                for k, v in metrics.items():
                    combined_metrics[f"{prefix}_{val}/{k}"] = v

        return total_loss, combined_metrics

    def fit(self, *args, **kwargs):
        if "dataset" in kwargs.keys():
            if type(kwargs["dataset"]) is SimulationOutput:
                kwargs["dataset"] = kwargs["dataset"].data

        return super(GraphicalApproximator, self).fit(*args, **kwargs, adapter=self.adapter)

    def sample(self, *, num_samples: int, conditions: Mapping[str, np.ndarray]) -> Mapping[str, np.ndarray]:
        summary_outputs = summary_outputs_by_network(self, conditions)
        batch_size = keras.ops.shape(summary_outputs[0])[0]
        data_node = self.graph.simulation_graph.data_node()
        variable_names = self.graph.simulation_graph.variable_names()

        inference_conditions = {}

        # add num_samples as repeats across the batch dimension
        for name in variable_names[data_node]:
            inference_conditions[name] = keras.ops.repeat(conditions[name], num_samples, axis=0)

        for i, inference_network in enumerate(self.inference_networks):
            cond = prepare_inference_conditions(self, inference_conditions, i)
            samples = inference_network.sample((batch_size * num_samples,), conditions=cond)
            split_output = split_network_output(self, samples, i)

            for k, v in split_output.items():
                inference_conditions[k] = v

        # build sample dict with introduced num_samples dimension at axis 1
        sample_dict = {}
        for k, v in inference_conditions.items():
            if k not in variable_names[data_node]:
                target_shape = (batch_size, num_samples, *keras.ops.shape(v)[1:])
                sample_dict[k] = keras.ops.convert_to_numpy(keras.ops.reshape(v, target_shape))

        return sample_dict

    def _batch_size_from_data(self, data):
        data_shapes = self.data_shapes(data)
        batch_size = next(iter(data_shapes.values()))[0]

        return batch_size

    def _data_shapes(self, adapted_data: SimulationOutput | Mapping) -> Mapping:
        if isinstance(adapted_data, dict):
            return keras.tree.map_structure(keras.ops.shape, adapted_data)
        elif isinstance(adapted_data, SimulationOutput):
            return keras.tree.map_structure(keras.ops.shape, adapted_data.data)

        return {}
