from typing import Literal

from tqdm.auto import tqdm

import numpy as np
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

    @staticmethod
    def resolve(
        summary_network: keras.Layer | None,
        inference_conditions: Tensor | None,
        summary_variables: Tensor | None,
        summary_output: Tensor | np.ndarray | None,
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
        summary_output : Tensor or None
            If already computed, the output of the summary network. If provided, this will be used instead of
            computing summaries again from summary variables.
        stage : str
            Current stage (``"training"``, ``"validation"``, or ``"inference"``).
        purpose : {"call", "metrics"}
            ``"call"``  — forward pass: returns raw summary outputs.
            ``"metrics"`` — training/validation: returns summary metric dict.
        batch_size : int, optional
            Batch size for the summary network (default is ``None``).
        **summary_kwargs
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

        if summary_variables is None and summary_output is None:
            raise ValueError("Summary variables are required when a summary network is present.")

        if purpose == "call":
            if summary_output is None:
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

                    batch_outputs = summary_network(
                        batch_variables, **filter_kwargs(batch_kwargs, summary_network.call)
                    )
                    batches.append(batch_outputs)

                summary_output = tree_concatenate(batches, axis=0)
            else:
                summary_output = keras.ops.convert_to_tensor(summary_output)
            conditions = concatenate_valid((inference_conditions, summary_output), axis=-1)
            return conditions, summary_output

        elif purpose == "metrics":
            metrics = summary_network.compute_metrics(
                summary_variables, stage=stage, **filter_kwargs(summary_kwargs, summary_network.compute_metrics)
            )
            summary_output = metrics.pop("outputs")
            conditions = concatenate_valid((inference_conditions, summary_output), axis=-1)
            return conditions, metrics

        else:
            raise ValueError(f"Unknown purpose={purpose!r}.")

    @staticmethod
    def resolve_ancestral(
        summary_network: keras.Layer | None,
        inference_conditions: Tensor | None,
        child_summary_variables: Tensor | None,
        n_datasets: int,
        n_children: int,
        n_parent_samples: int,
        batch_size: int | None = None,
        **summary_kwargs,
    ) -> tuple[Tensor | None, Tensor | None]:
        """Summarize child conditions before expansion, then concatenate with inference conditions.

        The summary network runs on ``n_datasets * n_children`` samples rather than
        ``n_datasets * n_children * n_parent_samples``, avoiding redundant forward passes.
        The resulting summaries are then expanded along the parent-sample axis and
        concatenated with the (already-expanded) inference conditions.

        Parameters
        ----------
        summary_network : keras.Layer or None
            The summary network.  If ``None``, ``child_summary_variables`` must also
            be ``None`` and ``inference_conditions`` is returned as-is.
        inference_conditions : Tensor or None, shape (flat_batch, ic_dim)
            Already-expanded inference conditions from the merged adapter call.
        child_summary_variables : Tensor or None, shape (n_datasets * n_children, sv_dim)
            Un-expanded child summary variables to be summarized before expansion.
        n_datasets : int
            Number of independent datasets.
        n_children : int
            Number of child conditions per dataset.
        n_parent_samples : int
            Number of parent samples per dataset.
        batch_size : int, optional
            Batch size for the summary network forward pass.
        **summary_kwargs
            Extra keyword arguments forwarded to the summary network.

        Returns
        -------
        resolved_conditions : Tensor or None
            ``inference_conditions`` concatenated with expanded summary outputs.
        summary_outputs : Tensor or None
            Raw summary network outputs at child level ``(n_datasets * n_children, summary_dim)``,
            or ``None`` if no summary network.
        """
        if summary_network is None:
            if child_summary_variables is not None:
                raise ValueError("Cannot use summary_variables without a summary network.")
            return inference_conditions, None

        if child_summary_variables is None:
            raise ValueError("Summary variables are required when a summary network is present.")

        flat_child_batch = n_datasets
        if batch_size is None:
            batch_size = flat_child_batch

        batches = []
        for i in tqdm(range(0, flat_child_batch, batch_size), desc="Summarizing", unit="batch"):
            batch_variables = slice_maybe_nested(child_summary_variables, i, i + batch_size)
            batch_kwargs = {
                k: slice_maybe_nested(v, i, i + batch_size) if hasattr(v, "shape") else v
                for k, v in summary_kwargs.items()
            }
            batch_outputs = summary_network(batch_variables, **filter_kwargs(batch_kwargs, summary_network.call))
            batches.append(batch_outputs)

        child_summaries = tree_concatenate(batches, axis=0)  # (n_datasets * n_children, summary_dim)

        # (n_datasets * n_children, summary_dim) -> (n_datasets, n_children, n_parent_samples, summary_dim)
        # -> (flat_batch, summary_dim)
        flat_batch = n_datasets * n_children * n_parent_samples
        child_summaries = keras.ops.reshape(child_summaries, (n_datasets, n_children, -1))
        expanded = keras.ops.expand_dims(child_summaries, axis=2)
        expanded = keras.ops.repeat(expanded, n_parent_samples, axis=2)
        expanded = keras.ops.reshape(expanded, (flat_batch, -1))

        conditions = concatenate_valid((inference_conditions, expanded), axis=-1)
        return conditions, child_summaries

    @staticmethod
    def get_config() -> dict:
        return {}

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ConditionBuilder":
        return cls(**deserialize(config, custom_objects=custom_objects))
