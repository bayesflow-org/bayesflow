import keras
import numpy as np
from keras.saving import (
    register_keras_serializable as serializable,
)

from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs, split_arrays, squeeze_inner_estimates_dict
from .continuous_approximator import ContinuousApproximator


@serializable(package="bayesflow.approximators")
class PointApproximator(ContinuousApproximator):
    """
    A workflow for fast amortized point estimation of a conditional distribution.

    The distribution is approximated by point estimators, parameterized by a feed-forward `PointInferenceNetwork`.
    Conditions can be compressed by an optional `SummaryNetwork` or used directly as input to the inference network.
    """

    def estimate(
        self,
        conditions: dict[str, np.ndarray],
        split: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray]]:
        if not self.built:
            raise AssertionError("PointApproximator needs to be built before predicting with it.")

        # Prepare the input conditions.
        conditions = self._prepare_conditions(conditions, **kwargs)
        # Run the internal estimation and convert the output to numpy.
        estimates = self._run_inference(conditions, **kwargs)
        # Postprocess the inference output with the inverse adapter.
        estimates = self._apply_inverse_adapter(estimates, **kwargs)
        # Optionally split the arrays along the last axis.
        if split:
            estimates = split_arrays(estimates, axis=-1)
        # Reorder the nested dictionary so that original variable names are at the top.
        estimates = self._reorder_estimates(estimates)
        # Remove unnecessary nesting.
        estimates = self._squeeze_estimates(estimates)

        return estimates

    def _prepare_conditions(self, conditions: dict[str, np.ndarray], **kwargs) -> dict[str, Tensor]:
        """Adapts and converts the conditions to tensors."""
        conditions = self.adapter(conditions, strict=False, stage="inference", **kwargs)
        return keras.tree.map_structure(keras.ops.convert_to_tensor, conditions)

    def _run_inference(self, conditions: dict[str, Tensor], **kwargs) -> dict[str, dict[str, np.ndarray]]:
        """Runs the internal _estimate function and converts the result to numpy arrays."""
        # Run the estimation.
        inference_output = self._estimate(**conditions, **kwargs)
        # Wrap the result in a dict and convert to numpy.
        wrapped_output = {"inference_variables": inference_output}
        return keras.tree.map_structure(keras.ops.convert_to_numpy, wrapped_output)

    def _apply_inverse_adapter(
        self, estimates: dict[str, dict[str, np.ndarray]], **kwargs
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        """Applies the inverse adapter on each inner element of the inference outputs."""
        processed = {}
        for score_key, score_val in estimates["inference_variables"].items():
            processed[score_key] = {}
            for head_key, estimate in score_val.items():
                adapted = self.adapter(
                    {"inference_variables": estimate},
                    inverse=True,
                    strict=False,
                    **kwargs,
                )
                processed[score_key][head_key] = adapted
        return processed

    def _reorder_estimates(
        self, estimates: dict[str, dict[str, dict[str, np.ndarray]]]
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:
        """Reorders the nested dictionary so that the inference variable names become the top-level keys."""
        # Grab the variable names from one sample inner dictionary.
        sample_inner = next(iter(next(iter(estimates.values())).values()))
        variable_names = sample_inner.keys()
        reordered = {}
        for variable in variable_names:
            reordered[variable] = {}
            for score_key, inner_dict in estimates.items():
                reordered[variable][score_key] = {inner_key: value[variable] for inner_key, value in inner_dict.items()}
        return reordered

    def _squeeze_estimates(
        self, estimates: dict[str, dict[str, dict[str, np.ndarray]]]
    ) -> dict[str, dict[str, np.ndarray]]:
        """Squeezes each inner estimate dictionary to remove unnecessary nesting."""
        squeezed = {}
        for variable, variable_estimates in estimates.items():
            squeezed[variable] = {
                score_key: squeeze_inner_estimates_dict(inner_estimate)
                for score_key, inner_estimate in variable_estimates.items()
            }
        return squeezed

    def _estimate(
        self,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        **kwargs,
    ) -> dict[str, dict[str, Tensor]]:
        if self.summary_network is None:
            if summary_variables is not None:
                raise ValueError("Cannot use summary variables without a summary network.")
        else:
            if summary_variables is None:
                raise ValueError("Summary variables are required when a summary network is present.")

            summary_outputs = self.summary_network(
                summary_variables, **filter_kwargs(kwargs, self.summary_network.call)
            )

            if inference_conditions is None:
                inference_conditions = summary_outputs
            else:
                inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=1)

        return self.inference_network(
            conditions=inference_conditions,
            **filter_kwargs(kwargs, self.inference_network.call),
        )
