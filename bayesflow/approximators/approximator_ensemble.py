from collections.abc import Mapping

import numpy as np

import keras

from bayesflow.types import Tensor


from .approximator import Approximator


class ApproximatorEnsemble(Approximator):
    def __init__(self, approximators: dict[str, Approximator], **kwargs):
        super().__init__(**kwargs)

        self.approximators = approximators

        self.num_approximators = len(self.approximators)

    def build_from_data(self, adapted_data: dict[str, any]):
        data_shapes = keras.tree.map_structure(keras.ops.shape, adapted_data)
        if len(data_shapes["inference_variables"]) > 2:
            # Remove the ensemble dimension from data_shapes. This expects data_shapes are the shapes of a
            # batch of training data, where the second axis corresponds to different approximators.
            data_shapes = {k: v[:1] + v[2:] for k, v in data_shapes.items()}
        self.build(data_shapes)

    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
        for approximator in self.approximators.values():
            approximator.build(data_shapes)

    def compute_metrics(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        sample_weight: Tensor = None,
        stage: str = "training",
    ) -> dict[str, dict[str, Tensor]]:
        # Prepare empty dict for metrics
        metrics = {}

        # Define the variable slices as None (default) or respective input
        _inference_variables = inference_variables
        _inference_conditions = inference_conditions
        _summary_variables = summary_variables
        _sample_weight = sample_weight

        for i, (approx_name, approximator) in enumerate(self.approximators.items()):
            # During training each approximator receives its own separate slice
            if stage == "training":
                # Pick out the correct slice for each ensemble member
                _inference_variables = inference_variables[:, i]
                if inference_conditions is not None:
                    _inference_conditions = inference_conditions[:, i]
                if summary_variables is not None:
                    _summary_variables = summary_variables[:, i]
                if sample_weight is not None:
                    _sample_weight = sample_weight[:, i]

            metrics[approx_name] = approximator.compute_metrics(
                inference_variables=_inference_variables,
                inference_conditions=_inference_conditions,
                summary_variables=_summary_variables,
                sample_weight=_sample_weight,
                stage=stage,
            )

        # Flatten metrics dict
        joint_metrics = {}
        for approx_name in metrics.keys():
            for metric_key, value in metrics[approx_name].items():
                joint_metrics[f"{approx_name}/{metric_key}"] = value
        metrics = joint_metrics

        # Sum over losses
        losses = [v for k, v in metrics.items() if "loss" in k]
        metrics["loss"] = keras.ops.sum(losses)

        return metrics

    def sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray]]:
        samples = {}
        for approx_name, approximator in self.approximators.items():
            if self._has_obj_method(approximator, "sample"):
                samples[approx_name] = approximator.sample(
                    num_samples=num_samples, conditions=conditions, split=split, **kwargs
                )
        return samples

    def log_prob(self, data: Mapping[str, np.ndarray], **kwargs) -> dict[str, np.ndarray]:
        log_prob = {}
        for approx_name, approximator in self.approximators.items():
            if self._has_obj_method(approximator, "log_prob"):
                log_prob[approx_name] = approximator.log_prob(data=data, **kwargs)
        return log_prob

    def estimate(
        self,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        estimates = {}
        for approx_name, approximator in self.approximators.items():
            if self._has_obj_method(approximator, "estimate"):
                estimates[approx_name] = approximator.estimate(conditions=conditions, split=split, **kwargs)
        return estimates

    def _has_obj_method(self, obj, name):
        method = getattr(obj, name, None)
        return callable(method)

    def _batch_size_from_data(self, data: Mapping[str, any]) -> int:
        """
        Fetches the current batch size from an input dictionary. Can only be used during training when
        inference variables as present.
        """
        return keras.ops.shape(data["inference_variables"])[0]
