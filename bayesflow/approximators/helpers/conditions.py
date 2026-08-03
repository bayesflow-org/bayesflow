from typing import Literal

from tqdm.auto import tqdm

import numpy as np
import keras

from bayesflow.types import Tensor
from bayesflow.utils.serialization import serializable, deserialize
from bayesflow.utils import (
    concatenate_valid,
    concatenate_valid_shapes,
    dim_maybe_nested,
    filter_kwargs,
    slice_maybe_nested,
    tree_concatenate,
)


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
        summary_outputs: Tensor | np.ndarray | None,
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
        summary_outputs : Tensor or None
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
        summary_outputs : Tensor, dict, or None
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

        if summary_variables is None and summary_outputs is None:
            raise ValueError("Summary variables are required when a summary network is present.")

        if purpose == "call":
            if summary_outputs is None:
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

                summary_outputs = tree_concatenate(batches, axis=0)
            else:
                summary_outputs = keras.ops.convert_to_tensor(summary_outputs)
            conditions = concatenate_valid((inference_conditions, summary_outputs), axis=-1)
            return conditions, summary_outputs

        elif purpose == "metrics":
            metrics = summary_network.compute_metrics(
                summary_variables, stage=stage, **filter_kwargs(summary_kwargs, summary_network.compute_metrics)
            )
            summary_outputs = metrics.pop("outputs")
            conditions = concatenate_valid((inference_conditions, summary_outputs), axis=-1)
            return conditions, metrics

        else:
            raise ValueError(f"Unknown purpose={purpose!r}.")

    @staticmethod
    def resolve_ancestral(
        summary_network: keras.Layer | None,
        inference_conditions: Tensor | None,
        child_summary_variables: Tensor | None,
        summary_outputs: Tensor | np.ndarray | None,
        num_datasets: int,
        num_children: int,
        num_parent_samples: int,
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
        inference_conditions : Tensor or None, shape (flat_batch, dim)
            Already-expanded inference conditions from the merged adapter call.
        child_summary_variables : Tensor or None, shape (num_datasets * num_children, sv_dim)
            Un-expanded child summary variables to be summarized before expansion.
        summary_outputs : Tensor or None
            If already computed, the output of the summary network. If provided, this will be used instead of
            computing summaries again from summary variables.
        num_datasets : int
            Number of independent datasets.
        num_children : int
            Number of child conditions per dataset.
        num_parent_samples : int
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

        total_datasets = num_datasets * num_children
        if batch_size is None:
            batch_size = total_datasets

        if summary_outputs is None:
            batches = []
            for i in tqdm(range(0, total_datasets, batch_size), desc="Summarizing", unit="batch"):
                batch_variables = slice_maybe_nested(child_summary_variables, i, i + batch_size)
                batch_kwargs = {
                    k: slice_maybe_nested(v, i, i + batch_size) if hasattr(v, "shape") else v
                    for k, v in summary_kwargs.items()
                }
                batch_outputs = summary_network(batch_variables, **filter_kwargs(batch_kwargs, summary_network.call))
                batches.append(batch_outputs)

            child_summaries = tree_concatenate(batches, axis=0)
            child_summaries = keras.ops.reshape(child_summaries, (num_datasets, num_children, -1))
        else:
            child_summaries = keras.ops.convert_to_tensor(summary_outputs)

        # (num_datasets * num_children, summary_dim) ->
        # (num_datasets, num_children, num_parent_samples, summary_dim)
        # -> (flat_batch, summary_dim)
        flat_batch = num_datasets * num_children * num_parent_samples
        expanded = keras.ops.expand_dims(child_summaries, axis=2)
        expanded = keras.ops.repeat(expanded, num_parent_samples, axis=2)
        expanded = keras.ops.reshape(expanded, (flat_batch, -1))

        conditions = concatenate_valid((inference_conditions, expanded), axis=-1)
        return conditions, child_summaries

    @staticmethod
    def get_config() -> dict:
        return {}

    @classmethod
    def from_config(cls, config: dict, custom_objects=None) -> "ConditionBuilder":
        return cls(**deserialize(config, custom_objects=custom_objects))


@serializable("bayesflow.approximators")
class AutoregressiveConditionBuilder(ConditionBuilder):
    """Resolve encoder inputs and causal decoder conditions for joint smoothing."""

    @staticmethod
    def encoder_input_shape(summary_shape: tuple, conditions_shape: tuple | None) -> tuple:
        if conditions_shape is not None and len(conditions_shape) < len(summary_shape):
            conditions_shape = tuple(summary_shape[:-1]) + (conditions_shape[-1],)
        return concatenate_valid_shapes((summary_shape, conditions_shape), axis=-1)

    @staticmethod
    def resolve_encoder_inputs(
        standardizer: keras.Layer,
        inference_conditions: Tensor | None,
        summary_variables: Tensor,
        *,
        stage: str,
        summary_mask: Tensor | None = None,
    ) -> Tensor:
        summary_variables = standardizer.maybe_standardize(
            summary_variables,
            key="summary_variables",
            stage=stage,
            mask=summary_mask,
        )
        inference_conditions = standardizer.maybe_standardize(
            inference_conditions,
            key="inference_conditions",
            stage=stage,
        )

        if inference_conditions is not None and len(inference_conditions.shape) < len(summary_variables.shape):
            inference_conditions = keras.ops.expand_dims(inference_conditions, axis=1)
            inference_conditions = keras.ops.broadcast_to(
                inference_conditions,
                (*keras.ops.shape(summary_variables)[:-1], keras.ops.shape(inference_conditions)[-1]),
            )

        return concatenate_valid((summary_variables, inference_conditions), axis=-1)

    @staticmethod
    def resolve_decoder_time(encoder_network: keras.Layer, encoder_inputs: Tensor) -> Tensor | None:
        """Return the explicit time vector consumed by the encoder, if one is configured."""
        time_axis = getattr(encoder_network, "time_axis", None)
        if time_axis is None:
            return None
        return encoder_inputs[..., time_axis]

    def resolve(
        self,
        *,
        standardizer: keras.Layer,
        encoder_network: keras.Layer,
        decoder_network: keras.Layer,
        inference_variables: Tensor | None,
        inference_conditions: Tensor | None,
        summary_variables: Tensor,
        stage: str,
        summary_attention_mask: Tensor | None = None,
        summary_mask: Tensor | None = None,
        inference_attention_mask: Tensor | None = None,
        inference_mask: Tensor | None = None,
        return_decoder_time: bool = False,
    ) -> tuple[Tensor | None, Tensor] | tuple[Tensor | None, Tensor, Tensor | None]:
        encoder_inputs = self.resolve_encoder_inputs(
            standardizer,
            inference_conditions,
            summary_variables,
            stage=stage,
            summary_mask=summary_mask,
        )
        decoder_time = self.resolve_decoder_time(encoder_network, encoder_inputs)
        encoder_kwargs = filter_kwargs(
            {
                "attention_mask": summary_attention_mask,
                "mask": summary_mask,
            },
            encoder_network.call,
        )
        encoder_kwargs = {key: value for key, value in encoder_kwargs.items() if value is not None}
        encoder_outputs = encoder_network(
            encoder_inputs,
            training=stage == "training",
            **encoder_kwargs,
        )

        if inference_variables is None:
            if return_decoder_time:
                return None, encoder_outputs, decoder_time
            return None, encoder_outputs

        decoder_kwargs = filter_kwargs(
            {
                "time": decoder_time,
                "target_mask": inference_mask,
                "encoder_mask": summary_mask,
                "attention_mask": inference_attention_mask,
                "training": stage == "training",
            },
            decoder_network.call,
        )
        conditions = decoder_network(inference_variables, encoder_outputs, **decoder_kwargs)

        if return_decoder_time:
            return conditions, encoder_outputs, decoder_time
        return conditions, encoder_outputs
