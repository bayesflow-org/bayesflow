import inspect
from collections.abc import Sequence, Callable, Mapping
from functools import partial
from typing import Literal, Tuple

import keras
import numpy as np

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils import split_arrays
from bayesflow.utils.serialization import serializable
from .continuous_approximator import ContinuousApproximator


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

        # prepare score computation
        if compute_prior_score is None:
            compute_prior_score_pre = None
        else:
            compute_prior_score_pre = partial(
                prepare_compute_prior_score,
                compute_prior_score=compute_prior_score,
                adapter=self.adapter,
                standardizer=self.standardizer,
            )

        inference_kwargs = kwargs | self._collect_mask_kwargs(self._INFERENCE_MASK_KEYS, adapted)
        inference_kwargs["compute_prior_score"] = compute_prior_score_pre
        if sample_shape == "infer":  # infer method cannot handle the compositional dimensions
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
            n_datasets, n_comp = keras.ops.shape(summary_outputs)[:2]
            summary_outputs = keras.ops.reshape(
                summary_outputs, (n_datasets * n_comp,) + keras.ops.shape(summary_outputs)[2:]
            )
            flattened_conditions = None
        elif conditions is not None:
            original_shapes = {}
            flattened_conditions = {}
            for key, value in conditions.items():  # Flatten compositional dimensions
                original_shapes[key] = value.shape
                n_datasets, n_comp = value.shape[:2]
                flattened_shape = (n_datasets * n_comp,) + value.shape[2:]
                flattened_conditions[key] = value.reshape(flattened_shape)
            n_datasets, n_comp = original_shapes[next(iter(original_shapes))][:2]
        else:
            raise ValueError(
                "At least one of 'conditions' or 'summary_outputs' must be provided for compositional sampling."
            )

        if n_comp <= 1:
            raise ValueError(
                "At least two conditioning variables are required for compositional sampling, got "
                f"{n_comp}. Use 'sample' instead."
            )

        resolved_conditions, adapted, summary_outputs = self._prepare_conditions(
            data=flattened_conditions,
            summary_outputs=summary_outputs,
            batch_size=batch_size,
            **kwargs,
        )

        # Reshape tensors back to (n_datasets, n_compositional, ...)
        resolved_conditions = keras.ops.reshape(
            resolved_conditions,
            (
                n_datasets,
                n_comp,
            )
            + keras.ops.shape(resolved_conditions)[1:],
        )
        if summary_outputs is not None:
            summary_outputs = keras.ops.reshape(
                summary_outputs,
                (
                    n_datasets,
                    n_comp,
                )
                + keras.ops.shape(summary_outputs)[1:],
            )
        return resolved_conditions, adapted, summary_outputs


def prepare_compute_prior_score(
    samples: Tensor,
    time: Tensor,
    compute_prior_score: Callable[[dict[str, np.ndarray], Tensor | None], dict[str, np.ndarray]],
    adapter: Adapter,
    standardizer,
) -> Tensor:
    samples = keras.tree.map_structure(
        lambda s: standardizer.maybe_standardize(s, key="inference_variables", stage="inference", forward=False),
        samples,
    )
    if keras.backend.backend() != "torch":  # samples cannot be converted to numpy, otherwise it breaks
        adapted_samples, log_det_jac = keras.tree.map_structure(
            lambda s: adapter({"inference_variables": s}, inverse=True, strict=False, log_det_jac=True),
            samples,
        )
    else:  # samples need to be converted to numpy, adapter cannot use torch tensors
        adapted_samples, log_det_jac = keras.tree.map_structure(
            lambda s: adapter(
                {"inference_variables": keras.ops.convert_to_numpy(s)}, inverse=True, strict=False, log_det_jac=True
            ),
            samples,
        )

    if len(log_det_jac) > 0:
        problematic_keys = [key for key in log_det_jac if log_det_jac[key] != 0.0]
        raise NotImplementedError(
            f"Cannot use compositional sampling with adapters "
            f"that have non-zero log_det_jac. Problematic keys: {problematic_keys}"
        )
    pass_time = len(inspect.signature(compute_prior_score).parameters) == 2
    if pass_time:
        prior_score = compute_prior_score(adapted_samples, time)
    else:
        prior_score = compute_prior_score(adapted_samples)

    for key in adapted_samples:
        prior_score[key] = keras.ops.cast(prior_score[key], "float32")

    prior_score = keras.tree.map_structure(keras.ops.convert_to_tensor, prior_score)
    out = keras.ops.concatenate([prior_score[key] for key in adapted_samples], axis=-1)
    if not pass_time:
        out = (1 - time) * out

    if "inference_variables" in standardizer.standardize:
        # Apply Jacobian correction from standardization
        # For standardization T^{-1}(z) = z * std + mean, the Jacobian is diagonal with std on diagonal
        # The gradient of log|det(J)| w.r.t. z is 0 since log|det(J)| = sum(log(std)) is constant w.r.t. z
        # But we need to transform the score: score_z = score_x * std where x = T^{-1}(z)
        standardize_layer = standardizer.standardize_layers["inference_variables"]

        # Compute the correct standard deviation for all components
        std_components = []
        for idx in range(len(standardize_layer.moving_mean)):
            std_val = standardize_layer.moving_std(idx)
            std_components.append(std_val)

        # Concatenate std components to match the shape of out
        if len(std_components) == 1:
            std = std_components[0]
        else:
            std = keras.ops.concatenate(std_components, axis=-1)

        # Expand std to match batch dimension of out
        std_expanded = keras.ops.expand_dims(std, 0)  # Add batch dimensions

        # Apply the Jacobian: score_z = score_x * std
        out = out * std_expanded
    return out
