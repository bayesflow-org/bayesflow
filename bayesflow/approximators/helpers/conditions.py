from typing import Literal

from tqdm.auto import tqdm

import keras

from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable, deserialize
from bayesflow.utils import concatenate_valid, dim_maybe_nested, filter_kwargs, slice_maybe_nested, tree_concatenate


@serializable("bayesflow.approximators")
class ConditionBuilder:
    """Resolves inference conditions and optional summary network outputs.

    Manages the logic for combining raw inference conditions with
    summary network outputs (if present) into a single conditions
    tensor.  Used by all approximators to keep the condition
    preparation pipeline consistent.
    """

    def __init__(self):
        super().__init__()

    def resolve(
        self,
        summary_network: keras.Layer | None,
        inference_conditions: Tensor | None,
        summary_variables: Tensor | None,
        stage: str,
        purpose: Literal["call", "metrics"],
        batch_size: int | None,
        **summary_kwargs,
    ):
        """Resolve inference conditions, optionally incorporating summary network outputs.

        When a summary network is present, it is called (``purpose="call"``) or
        its ``compute_metrics`` method is invoked (``purpose="metrics"``).  The
        resulting summary outputs are concatenated with ``inference_conditions``
        along the last axis to produce the final resolved conditions tensor.

        Parameters
        ----------
        summary_network : keras.Layer or None
            The summary network.  If ``None``, ``summary_variables`` must also
            be ``None`` and ``inference_conditions`` is returned as-is.
        inference_conditions : Tensor or None
            Conditioning variables for the inference network.
        summary_variables : Tensor or None
            Input tensor(s) for the summary network.  Required when
            ``summary_network`` is not ``None``.
        stage : str
            Current stage (``"training"``, ``"validation"``, or ``"inference"``).
        purpose : {"call", "metrics"}
            ``"call"``  — forward pass: returns raw summary outputs.
            ``"metrics"`` — training/validation: returns summary metric dict.
        batch_size : int, optional
            Batch size for the summary network (default is ``None``).
        **kwargs
            Extra keyword arguments forwarded to ``summary_network.call``
            (when ``purpose="call"``) or ``summary_network.compute_metrics``
            (when ``purpose="metrics"``).  Filtered via :func:`filter_kwargs`
            so only accepted parameters are passed (e.g. ``attention_mask``).

        Returns
        -------
        resolved_conditions : Tensor or None
            ``inference_conditions`` concatenated with summary outputs (if any).
        summary_output : Tensor, dict, or None
            ``purpose="call"``:  summary network output tensor, or ``None``.
            ``purpose="metrics"``: dict of summary metrics (may be empty).

        Raises
        ------
        ValueError
            If ``summary_variables`` is provided without a ``summary_network``,
            or vice-versa, or if ``purpose`` is unrecognised.
        """
        if summary_network is None:
            if summary_variables is not None:
                raise ValueError("Cannot use summary_variables without a summary network.")
            if purpose == "call":
                return inference_conditions, None
            else:
                return inference_conditions, {}

        if summary_variables is None:
            raise ValueError("Summary variables are required when a summary network is present.")

        if purpose == "call":
            batches = []
            num_conditions = dim_maybe_nested(summary_variables, axis=0)
            if batch_size is None:
                batch_size = num_conditions

            for i in tqdm(range(0, num_conditions, batch_size), desc="Summarizing", unit="batch"):
                batch_variables = slice_maybe_nested(summary_variables, i, i + batch_size)
                batch_kwargs = {
                    k: slice_maybe_nested(v, i, i + batch_size) if hasattr(v, "shape") else v
                    for k, v in summary_kwargs.items()
                }

                batch_outputs = summary_network(batch_variables, **filter_kwargs(batch_kwargs, summary_network.call))
                batches.append(batch_outputs)

            outputs = tree_concatenate(batches, axis=0)
            conditions = concatenate_valid((inference_conditions, outputs), axis=-1)
            return conditions, outputs

        elif purpose == "metrics":
            metrics = summary_network.compute_metrics(
                summary_variables, stage=stage, **filter_kwargs(summary_kwargs, summary_network.compute_metrics)
            )
            outputs = metrics.pop("outputs")
            conditions = concatenate_valid((inference_conditions, outputs), axis=-1)
            return conditions, metrics

        else:
            raise ValueError(f"Unknown purpose={purpose!r}.")

    def get_config(self) -> dict:
        return {}

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ConditionBuilder":
        return cls(**deserialize(config, custom_objects=custom_objects))
