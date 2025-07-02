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
        metrics = {}
        for approx_name, approximator in self.approximators.items():
            # TODO: actually do the slicing
            inference_variables_slice = inference_variables
            inference_conditions_slice = inference_conditions
            summary_variables_slice = summary_variables
            sample_weight_slice = sample_weight

            metrics[approx_name] = approximator.compute_metrics(
                inference_variables=inference_variables_slice,
                inference_conditions=inference_conditions_slice,
                summary_variables=summary_variables_slice,
                sample_weight=sample_weight_slice,
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
    ) -> dict[str, np.ndarray]:
        samples = {}
        for approx_name, approximator in self.approximators.items():
            if self._has_obj_method(approximator, "sample"):
                samples[approx_name] = approximator.sample(
                    num_samples=num_samples, conditions=conditions, split=split, **kwargs
                )
        return samples

    def _has_obj_method(self, obj, name):
        method = getattr(obj, name, None)
        return callable(method)

    def _batch_size_from_data(self, data: Mapping[str, any]) -> int:
        """
        Fetches the current batch size from an input dictionary. Can only be used during training when
        inference variables as present.
        """
        return keras.ops.shape(data["inference_variables"])[0]
