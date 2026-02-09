from collections.abc import Mapping, Sequence
from typing import Literal, Tuple

import numpy as np

import keras

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils import (
    logging,
    filter_kwargs,
    split_arrays,
    concatenate_valid_shapes,
)
from bayesflow.utils.serialization import serialize, deserialize, serializable

from .approximator import Approximator
from ..networks.standardization import Standardization

from _runtime import Sampler, ConditionBuilder


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
    adapter : bayesflow.adapters.Adapter
        Adapter for data processing. You can use :py:meth:`build_adapter`
        to create it.
    inference_network : InferenceNetwork
        The inference network used for posterior or likelihood approximation.
    summary_network : SummaryNetwork, optional
        The summary network used for data summarization (default is None).
    standardize : str | Sequence[str] | None
        The variables to standardize before passing to the networks. Can be either
        "all" or any subset of ["inference_variables", "summary_variables", "inference_conditions"].
        (default is "all").
    **kwargs : dict, optional
        Additional arguments passed to the :py:class:`bayesflow.approximators.Approximator` class.
    """

    def __init__(
        self,
        *,
        adapter: Adapter,
        inference_network: InferenceNetwork,
        summary_network: SummaryNetwork = None,
        standardize: str | Sequence[str] | None = "all",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.adapter = adapter
        self.inference_network = inference_network
        self.summary_network = summary_network
        self.sampler = Sampler()
        self.standardizer = Standardization(standardize)
        self.condition_builder = ConditionBuilder()

    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
        if self.summary_network is not None and not self.summary_network.built:
            self.summary_network.build(data_shapes["summary_variables"])

        if self.summary_network is not None:
            summary_outputs_shape = self.summary_network.compute_output_shape(data_shapes["summary_variables"])
        else:
            summary_outputs_shape = None

        if not self.inference_network.built:
            inference_conditions_shape = concatenate_valid_shapes(
                [data_shapes.get("inference_conditions"), summary_outputs_shape], axis=-1
            )
            self.inference_network.build(data_shapes["inference_variables"], inference_conditions_shape)

        if not self.standardizer.built:
            self.standardizer.build(data_shapes)

        self.built = True

    @classmethod
    def build_adapter(
        cls,
        inference_variables: Sequence[str],
        inference_conditions: Sequence[str] = None,
        summary_variables: Sequence[str] = None,
        sample_weight: str = None,
    ) -> Adapter:
        """Create an :py:class:`~bayesflow.adapters.Adapter` suited for the approximator.

        Parameters
        ----------
        inference_variables : Sequence of str
            Names of the inference variables (to be modeled) in the data dict.
        inference_conditions : Sequence of str, optional
            Names of the inference conditions (to be used as direct conditions) in the data dict.
        summary_variables : Sequence of str, optional
            Names of the summary variables (to be passed to a summary network) in the data dict.
        sample_weight : str, optional
            Name of the sample weights
        """

        adapter = Adapter()
        adapter.to_array()
        adapter.convert_dtype("float64", "float32")
        adapter.concatenate(inference_variables, into="inference_variables")

        if inference_conditions is not None:
            adapter.concatenate(inference_conditions, into="inference_conditions")

        if summary_variables is not None:
            adapter.as_set(summary_variables)
            adapter.concatenate(summary_variables, into="summary_variables")

        if sample_weight is not None:
            adapter = adapter.rename(sample_weight, "sample_weight")

        adapter.keep(["inference_variables", "inference_conditions", "summary_variables", "sample_weight"])

        return adapter

    def compile(
        self,
        *args,
        inference_metrics: Sequence[keras.Metric] = None,
        summary_metrics: Sequence[keras.Metric] = None,
        **kwargs,
    ):
        if inference_metrics:
            self.inference_network._metrics = inference_metrics

        if summary_metrics:
            if self.summary_network is None:
                logging.warning("Ignoring summary metrics because there is no summary network.")
            else:
                self.summary_network._metrics = summary_metrics

        return super().compile(*args, **kwargs)

    def build_from_data(self, adapted_data: dict[str, any]):
        self.build(keras.tree.map_structure(keras.ops.shape, adapted_data))

    def compile_from_config(self, config):
        self.compile(**deserialize(config))
        if hasattr(self, "optimizer") and self.built:
            # Create optimizer variables.
            self.optimizer.build(self.trainable_variables)

    def compute_metrics(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        sample_weight: Tensor = None,
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

        inference_variables = self.standardizer.standardize(inference_variables, key="inference_variables", stage=stage)
        inference_conditions = self.standardizer.standardize(
            inference_conditions, key="inference_conditions", stage=stage
        )
        summary_variables = self.standardizer.standardize(summary_variables, key="summary_variables", stage=stage)

        summary_metrics, conditions = self.condition_builder.resolve(
            self.summary_network, inference_conditions, summary_variables, stage=stage, purpose="metrics"
        )

        inference_metrics = self.inference_network.compute_metrics(
            inference_variables, conditions=conditions, sample_weight=sample_weight, stage=stage
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
            Additional keyword arguments passed to `keras.Model.fit()`, including (see also `build_dataset`):

            batch_size : int or None, default='auto'
                Number of samples per gradient update. Do not specify if `dataset` is provided as a
                `keras.utils.PyDataset`, `tf.data.Dataset`, `torch.utils.data.DataLoader`, or a generator function.
            epochs : int, default=1
                Number of epochs to train the model.
            verbose : {"auto", 0, 1, 2}, default="auto"
                Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
            callbacks : list of keras.callbacks.Callback, optional
                List of callbacks to apply during training.
            validation_split : float, optional
                Fraction of training data to use for validation (only supported if `dataset` consists of NumPy arrays
                or tensors).
            validation_data : tuple or dataset, optional
                Data for validation, overriding `validation_split`.
            shuffle : bool, default=True
                Whether to shuffle the training data before each epoch (ignored for dataset generators).
            initial_epoch : int, default=0
                Epoch at which to start training (useful for resuming training).
            steps_per_epoch : int or None, optional
                Number of steps (batches) before declaring an epoch finished.
            validation_steps : int or None, optional
                Number of validation steps per validation epoch.
            validation_batch_size : int or None, optional
                Number of samples per validation batch (defaults to `batch_size`).
            validation_freq : int, default=1
                Specifies how many training epochs to run before performing validation.

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

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        config = {
            "adapter": self.adapter,
            "inference_network": self.inference_network,
            "summary_network": self.summary_network,
        }

        return base_config | serialize(config)

    def get_compile_config(self):
        base_config = super().get_compile_config() or {}

        config = {
            "inference_metrics": self.inference_network._metrics,
            "summary_metrics": self.summary_network._metrics if self.summary_network is not None else None,
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

        adapted_data = self.adapter(conditions, strict=False)
        adapted_data = keras.tree.map_structure(keras.ops.convert_to_tensor, adapted_data)

        inference_conditions = adapted_data.get("inference_conditions")
        summary_variables = adapted_data.get("summary_variables")

        inference_conditions = self.standardizer.maybe_standardize(
            inference_conditions, key="inference_conditions", stage="inference"
        )
        summary_variables = self.standardizer.maybe_standardize(
            summary_variables, key="summary_variables", stage="inference"
        )

        conditions, summaries = self.condition_builder.resolve(
            self.summary_network, inference_conditions, summary_variables, stage="inference", purpose="call"
        )

        samples = self.sampler.sample(
            inference_network=self.inference_network,
            num_samples=num_samples,
            conditions=conditions,
            batch_size=batch_size,
            sample_shape=sample_shape,
            **filter_kwargs(kwargs, self.inference_network.sample),
        )

        samples = self.standardizer.maybe_standardize(
            samples, key="inference_conditions", stage="inference", forward=False
        )

        samples = keras.tree.map_structure(keras.ops.convert_to_numpy, {"inference_variables": samples})
        samples = self.adapter(samples, inverse=True, strict=False)

        if return_summaries and summaries is not None:
            samples["summaries"] = summaries

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

        adapted_data, log_det_jac = self.adapter(data, strict=False, log_det_jac=True, stage="inference")
        adapted_data = keras.tree.map_structure(keras.ops.convert_to_tensor, adapted_data)

        inference_conditions = adapted_data.get("inference_conditions")
        summary_variables = adapted_data.get("summary_variables")
        inference_variables = adapted_data.get("inference_variables")

        inference_conditions = self.standardizer.maybe_standardize(
            inference_conditions, key="inference_conditions", stage="inference"
        )
        summary_variables = self.standardizer.maybe_standardize(
            summary_variables, key="summary_variables", stage="inference"
        )
        inference_variables, log_det_jac_std = self.standardizer.maybe_standardize(
            inference_variables, key="inference_variables", stage="inference", log_det_jac=True
        )

        log_det_jac = log_det_jac.get("inference_variables", 0.0)
        log_det_jac += keras.ops.convert_to_numpy(log_det_jac_std)

        conditions, _ = self.condition_builder.resolve(
            self.summary_network, inference_conditions, summary_variables, stage="inference", purpose="call"
        )

        log_prob = self.inference_network.log_prob(
            inference_variables,
            conditions=conditions,
            **filter_kwargs(kwargs, self.inference_network.log_prob),
        )

        log_prob = keras.ops.convert_to_numpy(log_prob)

        log_prob = log_prob + log_det_jac

        return log_prob

    def summarize(self, conditions: Mapping[str, np.ndarray], **kwargs) -> np.ndarray:
        """
        Computes the learned summary statistics of given summary variables.

        The `data` dictionary is preprocessed using the `adapter` and passed through the summary network.

        Parameters
        ----------
        conditions : Mapping[str, np.ndarray]
            Dictionary of simulated or real quantities as NumPy arrays.
        **kwargs : dict
            Additional keyword arguments for the adapter and the summary network.

        Returns
        -------
        summaries : np.ndarray
            The learned summary statistics.
        """
        if self.summary_network is None:
            raise ValueError("A summary network is required to compute summaries.")

        adapted_data = self.adapter(conditions, strict=False)
        summary_variables = adapted_data.get("summary_variables")
        summary_variables = self.standardizer.maybe_standardize(
            summary_variables, key="summary_variables", stage="inference"
        )

        if summary_variables is None:
            raise ValueError("Summary variables are required to compute summaries.")

        summaries = self.summary_network(summary_variables, **filter_kwargs(kwargs, self.summary_network.call))
        summaries = keras.ops.convert_to_numpy(summaries)

        return summaries

    def _batch_size_from_data(self, data: Mapping[str, any]) -> int:
        """
        Fetches the current batch size from an input dictionary. Can only be used during training when
        inference variables as present.
        """
        return keras.ops.shape(data["inference_variables"])[0]
