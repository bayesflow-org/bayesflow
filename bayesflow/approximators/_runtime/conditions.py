from typing import Literal

import keras

from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable, deserialize
from bayesflow.utils import concatenate_valid, filter_kwargs


@serializable("bayesflow.approximators")
class ConditionBuilder:
    def __init__(self):
        super().__init__()

    def resolve(
        self,
        summary_network: keras.Layer | None,
        inference_conditions: Tensor | None,
        summary_variables: Tensor | None,
        stage: str,
        purpose: Literal["call", "metrics"],
        **kwargs,
    ):
        if summary_network is None:
            if summary_variables is not None:
                raise ValueError("Cannot use summary_variables without a summary network.")
            return (None, inference_conditions) if purpose == "call" else (inference_conditions, {})

        if summary_variables is None:
            raise ValueError("Summary variables are required when a summary network is present.")

        if purpose == "call":
            outputs = summary_network(summary_variables, **filter_kwargs(kwargs, summary_network.call))
            conditions = concatenate_valid((inference_conditions, outputs), axis=-1)
            return outputs, conditions

        if purpose == "metrics":
            metrics = summary_network.compute_metrics(summary_variables, stage=stage)
            outputs = metrics.pop("outputs")
            conditions = concatenate_valid((inference_conditions, outputs), axis=-1)
            return metrics, conditions

        raise ValueError(f"Unknown purpose={purpose!r}.")

    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ConditionBuilder":
        return cls(**deserialize(config, custom_objects=custom_objects))
