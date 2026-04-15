from collections.abc import Sequence, Callable, Mapping
from typing import Literal, Tuple


import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils import split_arrays
from bayesflow.utils.serialization import serializable

from .continuous_approximator import ContinuousApproximator
from .helpers import build_prior_score_fn


@serializable("bayesflow.approximators")
class CompositionalApproximator(ContinuousApproximator):
    """
    Defines a wrapper for estimating arbitrary compositional distributions of the form:
    `p(inference_variables | [summary(summary_variables), ...], inference_conditions)`

    Parameters
    ----------
    inference_network : InferenceNetwork
        The inference network used for posterior or likelihood approximation.
    adapter : bayesflow.adapters.Adapter, optional
        Adapter for data processing. You can use :py:meth:`build_adapter`
        to create it. If ``None`` (default), an identity adapter is used
        that makes a shallow copy and passes data through unchanged.
    summary_network : SummaryNetwork, optional
        The summary network used for data summarization (default is None).
    standardize : str | Sequence[str] | None
        The variables to standardize before passing to the networks. Can be either
        "all" or any subset of ["inference_variables", "summary_variables", "inference_conditions"].
        (default is "inference_variables").
    **kwargs : dict, optional
        Additional arguments passed to the :py:class:`bayesflow.approximators.Approximator` class.
    """

    def __init__(
        self,
        *,
        inference_network: InferenceNetwork,
        adapter: Adapter = None,
        summary_network: SummaryNetwork = None,
        standardize: str | Sequence[str] | None = "inference_variables",
        **kwargs,
    ):
        super().__init__(
            adapter=adapter,
            inference_network=inference_network,
            summary_network=summary_network,
            standardize=standardize,
            **kwargs,
        )

    def compositional_sample(
        self,
        *,
        num_samples: int,
        conditions: dict[str, np.ndarray] | None = None,
        compute_prior_score: Callable[[dict[str, np.ndarray], np.ndarray | None], dict[str, np.ndarray]] = None,
        split: bool = False,
        batch_size: int | None = None,
        sample_shape: Literal["infer"] | Tuple[int] | int = "infer",
        return_summaries: bool = False,
        summary_outputs: Tensor | np.ndarray | None = None,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Generates compositional samples from the approximator given input conditions. The `conditions` dictionary is
         preprocessed using the `adapter`. Samples are converted to NumPy arrays after inference.
         Expected shape of each condition variable is (n_datasets, n_compositional, ...), where n_compositional >= 2.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        conditions : dict[str, np.ndarray], optional
            Dictionary of conditioning variables as NumPy arrays.
        compute_prior_score : Callable[[dict[str, np.ndarray], np.ndarray | None], dict[str, np.ndarray]], optional
            A function that computes the score of the log prior distribution. Optionally, the function can have a time
            argument, otherwise the prior score is multiplied with (1-t), where t is diffusion time.
            If none provided, the unconditional score is used.
        split : bool, default=False
            Whether to split the output arrays along the last axis and return one sample array per target variable.
        batch_size : int or None, optional
            If provided, the conditions are split into batches of size `batch_size`, for which samples are generated
            sequentially. Can help with memory management for large sample sizes.
        sample_shape : str or tuple of int, optional
            Trailing structural dimensions of each generated sample, excluding the batch and target (intrinsic)
            dimension. For example, use `(time,)` for time series or `(height, width)` for images.

            If set to `"infer"` (default), the structural dimensions are inferred from the `inference_conditions`.
            In that case, all non-vector dimensions except the last (channel) dimension are treated as structural
            dimensions. For example, if the final `inference_conditions` have shape `(batch_size, time, channels)`,
            then `sample_shape` is inferred as `(time,)`, and the generated samples will have shape
            `(num_conditions, num_samples, time, target_dim)`.
        return_summaries: bool, optional
            If set to True and a summary network is present, will return the learned summary statistics for
            the provided conditions.
        summary_outputs : Tensor | np.ndarray | None, optional
            Precomputed summary outputs to be used as conditions for sampling. If provided, these will be used instead
            of the conditions. Should have shape (n_datasets, n_compositional_conditions, ...).
        **kwargs : dict
            Additional keyword arguments for the sampling process.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing generated samples with the same keys as `conditions`.
        """
        resolved_conditions, adapted, summary_outputs = self._prepare_compositional_conditions(
            conditions=conditions, batch_size=batch_size, summary_outputs=summary_outputs
        )

        if compute_prior_score is None:
            compute_prior_score_pre = None
        else:
            compute_prior_score_pre = build_prior_score_fn(
                compute_prior_score, adapter=self.adapter, standardizer=self.standardizer
            )

        inference_kwargs = kwargs | self._collect_mask_kwargs(self._INFERENCE_MASK_KEYS, adapted)
        inference_kwargs["compute_prior_score"] = compute_prior_score_pre

        # NOTE: infer option of sample cannot handle the compositional dimensions
        if sample_shape == "infer":
            sample_shape = tuple(keras.ops.shape(resolved_conditions)[2:-1])

        samples = self.sampler.sample(
            inference_network=self.inference_network,
            num_samples=num_samples,
            conditions=resolved_conditions,
            batch_size=batch_size,
            sample_shape=sample_shape,
            **inference_kwargs,
        )

        # Unstandardize and inverse-adapt samples (tree-aware for nested dict outputs)
        samples = keras.tree.map_structure(
            lambda s: self.standardizer.maybe_standardize(
                s, key="inference_variables", stage="inference", forward=False
            ),
            samples,
        )
        samples = keras.tree.map_structure(
            lambda s: self.adapter({"inference_variables": keras.ops.convert_to_numpy(s)}, inverse=True, strict=False),
            samples,
        )

        if return_summaries and summary_outputs is not None:
            samples["_summaries"] = summary_outputs

        if split:
            samples = split_arrays(samples, axis=-1)

        return samples

    def _prepare_compositional_conditions(
        self,
        conditions: Mapping[str, np.ndarray] | None,
        batch_size: int | None = None,
        summary_outputs: Tensor | np.ndarray | None = None,
        **kwargs,
    ) -> tuple[Tensor | None, dict[str, Tensor], Tensor | None]:
        if summary_outputs is not None:
            num_datasets, num_items = keras.ops.shape(summary_outputs)[:2]
            summary_outputs = keras.ops.reshape(
                summary_outputs, (num_datasets * num_items,) + keras.ops.shape(summary_outputs)[2:]
            )
            flattened_conditions = None

        elif conditions is not None:
            original_shapes = {}
            flattened_conditions = {}

            for key, value in conditions.items():
                original_shapes[key] = value.shape
                num_datasets, num_items = value.shape[:2]
                flattened_shape = (num_datasets * num_items,) + value.shape[2:]
                flattened_conditions[key] = value.reshape(flattened_shape)
            num_datasets, num_items = original_shapes[next(iter(original_shapes))][:2]

        else:
            raise ValueError(
                "At least one of 'conditions' or 'summary_outputs' must be provided for compositional sampling."
            )

        if num_items <= 1:
            raise ValueError(
                "At least two conditioning variables are required for compositional sampling, got "
                f"{num_items}. Use 'sample' instead."
            )

        resolved_conditions, adapted, summary_outputs = self._prepare_conditions(
            data=flattened_conditions,
            summary_outputs=summary_outputs,
            batch_size=batch_size,
            **kwargs,
        )

        # Reshape tensors back to (num_datasets, num_items, ...)
        restored_shape = (num_datasets, num_items) + keras.ops.shape(resolved_conditions)[1:]

        resolved_conditions = keras.ops.reshape(resolved_conditions, restored_shape)

        if summary_outputs is not None:
            summary_outputs = keras.ops.reshape(summary_outputs, restored_shape)

        return resolved_conditions, adapted, summary_outputs
