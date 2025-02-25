import keras
import numpy as np
from keras.saving import (
    register_keras_serializable as serializable,
)

from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs, split_arrays
from .continuous_approximator import ContinuousApproximator


@serializable(package="bayesflow.approximators")
class PointApproximator(ContinuousApproximator):
    """
    Defines a workflow for performing fast posterior or likelihood inference.
    The distribution is approximated by a point with an feed-forward network and an optional summary network.
    """

    def estimate(
        self,
        conditions: dict[str, np.ndarray],
        split: bool = False,
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray]]:
        if not self.built:
            raise AssertionError("PointApproximator needs to be built before predicting with it.")

        conditions = self.adapter(conditions, strict=False, stage="inference", **kwargs)
        conditions = keras.tree.map_structure(keras.ops.convert_to_tensor, conditions)
        conditions = {"inference_variables": self._estimate(**conditions)}
        conditions = keras.tree.map_structure(keras.ops.convert_to_numpy, conditions)
        conditions = {
            outer_key: {
                inner_key: self.adapter(
                    dict(inference_variables=conditions["inference_variables"][outer_key][inner_key]),
                    inverse=True,
                    strict=False,
                    **kwargs,
                )
                for inner_key in conditions["inference_variables"][outer_key].keys()
            }
            for outer_key in conditions["inference_variables"].keys()
        }

        if split:
            conditions = split_arrays(conditions, axis=-1)

        # get original variable names to reorder them to highest level
        inference_variable_names = next(iter(next(iter(conditions.values())).values())).keys()

        # change ordering of nested dictionary
        conditions = {
            variable_name: {
                outer_key: {
                    inner_key: conditions[outer_key][inner_key][variable_name]
                    for inner_key in conditions[outer_key].keys()
                }
                for outer_key in conditions.keys()
            }
            for variable_name in inference_variable_names
        }

        def squeeze_dict(d):
            if len(d.keys()) == 1 and "value" in d.keys():
                return d["value"]
            else:
                return d

        # remove unnecessary nesting
        conditions = {
            variable_name: {
                outer_key: squeeze_dict(conditions[variable_name][outer_key])
                for outer_key in conditions[variable_name].keys()
            }
            for variable_name in conditions.keys()
        }

        return conditions

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
            **filter_kwargs(kwargs, self.inference_network.sample),
        )
