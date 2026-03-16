from collections.abc import Mapping, Sequence, Callable
from typing import Literal, Tuple
from functools import partial

import numpy as np

import keras

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils import split_arrays
from bayesflow.utils.serialization import serialize, serializable

from .approximator import Approximator
from .helpers import Sampler, ConditionBuilder

from ..networks.helpers import Standardization


@serializable("bayesflow.approximators")
class ContinuousApproximator(Approximator):
    """
    Defines a wrapper for estimating arbitrary continuous distributions of the form:
    `p(inference_variables | summary(summary_variables), inference_conditions)`

    Any of the quantities on the RHS are optional. Can be used for neural posterior
    estimation (NPE), neural likelihood estimation (NLE), or any other kind of
    neural density estimation.

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
        super().__init__(**kwargs)
        self.adapter = adapter if adapter is not None else Adapter()
        self.inference_network = inference_network
        self.summary_network = summary_network
        self.sampler = Sampler()
        self.standardizer = Standardization(standardize)
        self.condition_builder = ConditionBuilder()
        self.has_distribution = True

    def compute_metrics(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        sample_weight: Tensor = None,
        summary_attention_mask: Tensor = None,
        summary_mask: Tensor = None,
        inference_attention_mask: Tensor = None,
        inference_mask: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:
        """
        Computes loss and tracks metrics for the inference and summary networks.

        This method orchestrates the end-to-end computation of metrics and loss for a model
        with both inference and optional summary network. It handles standardization of input
        variables, combines summary outputs with inference conditions when necessary, and
        aggregates loss and all tracked metrics into a unified dictionary. The returned dictionary
        includes both the total loss and all individual metrics, with keys indicating their source.

        Parameters
        ----------
        inference_variables : Tensor
            Input tensor(s) for the inference network. These are typically latent variables to be modeled.
        inference_conditions : Tensor, optional
            Conditioning variables for the inference network (default is None).
            May be combined with outputs from the summary network if present.
        summary_variables : Tensor, optional
            Input tensor(s) for the summary network (default is None). Required if
            a summary network is present.
        sample_weight : Tensor, optional
            Weighting tensor for metric computation (default is None).
        summary_attention_mask : Tensor, optional
            Attention mask forwarded to the summary network (default is None).
        summary_mask : Tensor, optional
            Padding / key mask forwarded to the summary network (default is None).
        inference_attention_mask : Tensor, optional
            Attention mask forwarded to the inference network (default is None).
        inference_mask : Tensor, optional
            Padding / key mask forwarded to the inference network (default is None).
        stage : str, optional
            Current training stage (e.g., "training", "validation", "inference"). Controls
            the behavior of standardization and some metric computations (default is "training").

        Returns
        -------
        metrics : dict[str, Tensor]
            Dictionary containing the total loss under the key "loss", as well as all tracked
            metrics for the inference and summary networks. Each metric key is prefixed with
            "inference_" or "summary_" to indicate its source.
        """

        inference_variables = self.standardizer.maybe_standardize(
            inference_variables, key="inference_variables", stage=stage
        )

        masks = dict(
            summary_attention_mask=summary_attention_mask,
            summary_mask=summary_mask,
            inference_attention_mask=inference_attention_mask,
            inference_mask=inference_mask,
        )
        summary_kwargs = self._collect_mask_kwargs(self._SUMMARY_MASK_KEYS, masks)
        inference_kwargs = self._collect_mask_kwargs(self._INFERENCE_MASK_KEYS, masks)

        resolved_conditions, summary_metrics = self._standardize_and_resolve(
            inference_conditions, summary_variables, stage=stage, purpose="metrics", **summary_kwargs
        )
        inference_metrics = self.inference_network.compute_metrics(
            inference_variables,
            conditions=resolved_conditions,
            sample_weight=sample_weight,
            stage=stage,
            **inference_kwargs,
        )

        if "loss" in summary_metrics:
            loss = inference_metrics["loss"] + summary_metrics["loss"]
        else:
            loss = inference_metrics.pop("loss")

        inference_metrics = {f"{key}/inference_{key}": value for key, value in inference_metrics.items()}
        summary_metrics = {f"{key}/summary_{key}": value for key, value in summary_metrics.items()}

        metrics = {"loss": loss} | inference_metrics | summary_metrics
        return metrics

    def fit(self, *args, **kwargs):
        """
        Trains the approximator on the provided dataset or on-demand data generated from the given simulator.
        If `dataset` is not provided, a dataset is built from the `simulator`.
        If the model has not been built, it will be built using a batch from the dataset.

        Parameters
        ----------
        dataset : keras.utils.PyDataset, optional
            A dataset containing simulations for training. If provided, `simulator` must be None.
        simulator : Simulator, optional
            A simulator used to generate a dataset. If provided, `dataset` must be None.
        **kwargs
            Additional keyword arguments passed to `keras.Model.fit()`, as described in:

        https://github.com/keras-team/keras/blob/v3.13.2/keras/src/backend/tensorflow/trainer.py#L314

        Returns
        -------
        keras.callbacks.History
            A history object containing the training loss and metrics values.

        Raises
        ------
        ValueError
            If both `dataset` and `simulator` are provided or neither is provided.
        """
        return super().fit(*args, **kwargs, adapter=self.adapter)

    def get_config(self):
        base_config = super().get_config()
        config = {
            "adapter": self.adapter,
            "inference_network": self.inference_network,
            "summary_network": self.summary_network,
            "standardize": self.standardizer.standardize,
        }

        return base_config | serialize(config)

    def sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        batch_size: int | None = None,
        sample_shape: Literal["infer"] | Tuple[int] | int = "infer",
        return_summaries: bool = False,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Generates samples from the approximator given input conditions. The `conditions` dictionary is preprocessed
        using the `adapter`. Samples are converted to NumPy arrays after inference.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        conditions : dict[str, np.ndarray]
            Dictionary of conditioning variables as NumPy arrays.
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
        **kwargs : dict
            Additional keyword arguments for the sampling process.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing generated samples with the same keys as `conditions`.
        """

        resolved_conditions, adapted, summary_outputs = self._prepare_conditions(conditions)

        inference_kwargs = kwargs | self._collect_mask_kwargs(self._INFERENCE_MASK_KEYS, adapted)

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

    def log_prob(self, data: Mapping[str, np.ndarray], **kwargs) -> np.ndarray:
        """
        Computes the log-probability of given data under the model. The `data` dictionary is preprocessed using the
        `adapter`. Log-probabilities are returned as NumPy arrays.

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            Dictionary of observed data as NumPy arrays.
        **kwargs : dict
            Additional keyword arguments for the adapter and log-probability computation.

        Returns
        -------
        np.ndarray
            Log-probabilities of the distribution `p(inference_variables | inference_conditions, h(summary_conditions))`
        """

        # NOTE: We cannot use _prepare_conditions here because we need
        # log_det_jac from the adapter call (log_det_jac=True), which
        # _prepare_conditions does not support.
        adapted, log_det_jac = self.adapter(data, strict=False, log_det_jac=True, stage="inference")
        adapted = keras.tree.map_structure(keras.ops.convert_to_tensor, adapted)

        summary_kwargs = self._collect_mask_kwargs(self._SUMMARY_MASK_KEYS, adapted)

        resolved_conditions, _ = self._standardize_and_resolve(
            adapted.get("inference_conditions"),
            adapted.get("summary_variables"),
            stage="inference",
            **summary_kwargs,
        )

        inference_variables, log_det_jac_std = self.standardizer.maybe_standardize(
            adapted.get("inference_variables"), key="inference_variables", stage="inference", log_det_jac=True
        )

        log_det_jac = log_det_jac.get("inference_variables", 0.0)
        log_det_jac += keras.ops.convert_to_numpy(log_det_jac_std)

        inference_kwargs = kwargs | self._collect_mask_kwargs(self._INFERENCE_MASK_KEYS, adapted)

        log_prob = self.inference_network.log_prob(
            inference_variables,
            conditions=resolved_conditions,
            **inference_kwargs,
        )

        log_prob = keras.tree.map_structure(keras.ops.convert_to_numpy, log_prob)
        log_prob = keras.tree.map_structure(lambda lp: lp + log_det_jac, log_prob)

        return log_prob

    def compositional_sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        compute_prior_score: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]],
        split: bool = False,
        batch_size: int | None = None,
        sample_shape: Literal["infer"] | Tuple[int] | int = "infer",
        return_summaries: bool = False,
        **kwargs,
    ) -> dict[str, np.ndarray]:
        """
        Generates samples from the approximator given input conditions. The `conditions` dictionary is preprocessed
        using the `adapter`. Samples are converted to NumPy arrays after inference.

        Parameters
        ----------
        num_samples : int
            Number of samples to generate.
        conditions : dict[str, np.ndarray]
            Dictionary of conditioning variables as NumPy arrays.
        compute_prior_score : Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]
            A function that computes the score of the log prior distribution.
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
        **kwargs : dict
            Additional keyword arguments for the sampling process.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing generated samples with the same keys as `conditions`.
        """
        original_shapes = {}
        flattened_conditions = {}
        for key, value in conditions.items():  # Flatten compositional dimensions
            original_shapes[key] = value.shape
            n_datasets, n_comp = value.shape[:2]
            flattened_shape = (n_datasets * n_comp,) + value.shape[2:]
            flattened_conditions[key] = value.reshape(flattened_shape)
        n_datasets, n_comp = original_shapes[next(iter(original_shapes))][:2]

        if n_comp <= 1:
            raise ValueError(
                "At least two conditioning variables are required for compositional sampling, got "
                f"{n_comp}. Use 'sample' instead."
            )

        resolved_conditions, adapted, summary_outputs = self._prepare_conditions(flattened_conditions)

        # Reshape conditions tensor back to (n_datasets, n_compositional, ...)
        resolved_conditions = keras.ops.reshape(
            resolved_conditions,
            (
                n_datasets,
                n_comp,
            )
            + keras.ops.shape(resolved_conditions)[1:],
        )
        # prepare score computation
        compute_prior_score_pre = partial(self._compute_prior_score_pre, compute_prior_score=compute_prior_score)

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

    def _compute_prior_score_pre(
        self, _samples: Tensor, compute_prior_score: Callable[[Mapping[str, np.ndarray]], Mapping[str, np.ndarray]]
    ) -> Tensor:
        _samples = keras.tree.map_structure(
            lambda s: self.standardizer.maybe_standardize(
                s, key="inference_variables", stage="inference", forward=False
            ),
            _samples,
        )
        if keras.backend.backend() != "torch":  # samples cannot be converted to numpy, otherwise it breaks
            adapted_samples, log_det_jac = keras.tree.map_structure(
                lambda s: self.adapter({"inference_variables": s}, inverse=True, strict=False, log_det_jac=True),
                _samples,
            )
        else:  # samples need to be converted to numpy, adapter cannot use torch tensors
            adapted_samples, log_det_jac = keras.tree.map_structure(
                lambda s: self.adapter(
                    {"inference_variables": keras.ops.convert_to_numpy(s)}, inverse=True, strict=False, log_det_jac=True
                ),
                _samples,
            )

        if len(log_det_jac) > 0:
            problematic_keys = [key for key in log_det_jac if log_det_jac[key] != 0.0]
            raise NotImplementedError(
                f"Cannot use compositional sampling with adapters "
                f"that have non-zero log_det_jac. Problematic keys: {problematic_keys}"
            )

        prior_score = compute_prior_score(adapted_samples)
        for key in adapted_samples:
            prior_score[key] = keras.ops.cast(prior_score[key], "float32")

        prior_score = keras.tree.map_structure(keras.ops.convert_to_tensor, prior_score)
        out = keras.ops.concatenate([prior_score[key] for key in adapted_samples], axis=-1)

        if "inference_variables" in self.standardizer.standardize:
            # Apply jacobian correction from standardization
            # For standardization T^{-1}(z) = z * std + mean, the jacobian is diagonal with std on diagonal
            # The gradient of log|det(J)| w.r.t. z is 0 since log|det(J)| = sum(log(std)) is constant w.r.t. z
            # But we need to transform the score: score_z = score_x * std where x = T^{-1}(z)
            standardize_layer = self.standardizer.standardize_layers["inference_variables"]

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

            # Apply the jacobian: score_z = score_x * std
            out = out * std_expanded
        return out
