from collections.abc import Mapping, Sequence, Callable

import numpy as np

import keras
import warnings

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils import (
    filter_kwargs,
    logging,
    split_arrays,
    squeeze_inner_estimates_dict,
    concatenate_valid,
    concatenate_valid_shapes,
)
from bayesflow.utils.serialization import serialize, deserialize, serializable

from .approximator import Approximator
from ..networks.standardization import Standardization


@serializable("bayesflow.approximators")
class ContinuousApproximator(Approximator):
    """
    Defines a workflow for performing fast posterior or likelihood inference.
    The distribution is approximated with an inference network and an optional summary network.

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

    CONDITION_KEYS = ["summary_variables", "inference_conditions"]

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

        if isinstance(standardize, str) and standardize != "all":
            self.standardize = [standardize]
        else:
            self.standardize = standardize or []

        if self.standardize == "all":
            # we have to lazily initialize these
            self.standardize_layers = None
        else:
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

    def build(self, data_shapes: dict[str, tuple[int] | dict[str, dict]]) -> None:
        # Build summary network and get output shape if present
        summary_outputs_shape = None
        if self.summary_network is not None:
            if not self.summary_network.built:
                self.summary_network.build(data_shapes["summary_variables"])
            summary_outputs_shape = self.summary_network.compute_output_shape(data_shapes["summary_variables"])

        # Compute inference_conditions_shape by combining original and summary outputs
        inference_conditions_shape = concatenate_valid_shapes(
            [data_shapes.get("inference_conditions"), summary_outputs_shape], axis=-1
        )

        # Build inference network if needed
        if not self.inference_network.built:
            self.inference_network.build(data_shapes["inference_variables"], inference_conditions_shape)

        # Set up standardization layers if requested
        if self.standardize == "all":
            # Only include variables present in data_shapes
            self.standardize = [
                var
                for var in ["inference_variables", "summary_variables", "inference_conditions"]
                if var in data_shapes
            ]
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

        # Build all standardization layers
        for var, layer in self.standardize_layers.items():
            layer.build(data_shapes[var])

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

        summary_metrics, summary_outputs = self._compute_summary_metrics(summary_variables, stage=stage)

        if "inference_conditions" in self.standardize:
            inference_conditions = self.standardize_layers["inference_conditions"](inference_conditions, stage=stage)
        inference_conditions = concatenate_valid((inference_conditions, summary_outputs), axis=-1)

        inference_variables = self._prepare_inference_variables(inference_variables, stage=stage)

        inference_metrics = self.inference_network.compute_metrics(
            inference_variables, conditions=inference_conditions, sample_weight=sample_weight, stage=stage
        )

        if "loss" in summary_metrics:
            loss = inference_metrics["loss"] + summary_metrics["loss"]
        else:
            loss = inference_metrics.pop("loss")

        inference_metrics = {f"{key}/inference_{key}": value for key, value in inference_metrics.items()}
        summary_metrics = {f"{key}/summary_{key}": value for key, value in summary_metrics.items()}

        metrics = {"loss": loss} | inference_metrics | summary_metrics
        return metrics

    def _compute_summary_metrics(self, summary_variables: Tensor | None, stage: str) -> tuple[dict, Tensor | None]:
        """Helper function to compute summary metrics and outputs."""
        if self.summary_network is None:
            if summary_variables is not None:
                raise ValueError("Cannot compute summaries from summary_variables without a summary network.")
            return {}, None

        if summary_variables is None:
            raise ValueError("Summary variables are required when a summary network is present.")

        if "summary_variables" in self.standardize:
            summary_variables = self.standardize_layers["summary_variables"](summary_variables, stage=stage)

        summary_metrics = self.summary_network.compute_metrics(summary_variables, stage=stage)
        summary_outputs = summary_metrics.pop("outputs")
        return summary_metrics, summary_outputs

    def _prepare_inference_variables(self, inference_variables: Tensor, stage: str) -> Tensor:
        """Helper function to convert inference variables to tensors and optionally standardize them."""
        if "inference_variables" in self.standardize:
            inference_variables = self.standardize_layers["inference_variables"](inference_variables, stage=stage)

        return inference_variables

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
            "standardize": self.standardize,
        }

        return base_config | serialize(config)

    def get_compile_config(self):
        base_config = super().get_compile_config() or {}

        config = {
            "inference_metrics": self.inference_network._metrics,
            "summary_metrics": self.summary_network._metrics if self.summary_network is not None else None,
        }

        return base_config | serialize(config)

    def estimate(
        self,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
        estimators: Mapping[str, Callable] = None,
        num_samples: int = 1000,
        **kwargs,
    ) -> dict[str, dict[str, np.ndarray]]:
        """
        Estimate summary statistics for variables based on given conditions.

        This function samples data using the object's ``sample`` method according to the provided
        conditions and then computes summary statistics for each variable using a set of estimator
        functions. By default, it calculates the mean, median, and selected quantiles (10th, 50th,
        and 90th percentiles). Users can also supply custom estimators that override or extend the
        default ones.

        Parameters
        ----------
        conditions : Mapping[str, np.ndarray]
            A mapping from variable names to numpy arrays representing the conditions under which
            samples should be generated.
        split : bool, optional
            If True, indicates that the data sampling process should split the samples based on an
            internal logic. The default is False.
        estimators : Mapping[str, Callable], optional
            A dictionary where keys are estimator names and values are callables. Each callable must
            accept an array and an axis parameter, and return a dictionary with the computed statistic.
            If not provided, a default set of estimators is used:
                - 'mean': Computes the mean along the specified axis.
                - 'median': Computes the median along the specified axis.
                - 'quantiles': Computes the 10th, 50th, and 90th percentiles along the specified axis,
                  then rearranges the axes for convenience.
        num_samples : int, optional
            The number of samples to generate for each variable. The default is 1000.
        **kwargs
            Additional keyword arguments passed to the ``sample`` method.

        Returns
        -------
        dict[str, dict[str, np.ndarray]]
            A nested dictionary where the outer keys correspond to variable names and the inner keys
            correspond to estimator names. Each inner dictionary contains the computed statistic(s) for
            the variable, potentially with reduced nesting via ``squeeze_inner_estimates_dict``.
        """

        estimators = estimators or {}
        estimators = (
            dict(
                mean=lambda x, axis: dict(value=np.mean(x, keepdims=True, axis=axis)),
                median=lambda x, axis: dict(value=np.median(x, keepdims=True, axis=axis)),
                quantiles=lambda x, axis: dict(value=np.moveaxis(np.quantile(x, q=[0.1, 0.5, 0.9], axis=axis), 0, 1)),
            )
            | estimators
        )

        samples = self.sample(num_samples=num_samples, conditions=conditions, split=split, **kwargs)

        estimates = {
            variable_name: {
                estimator_name: func(samples[variable_name], axis=1) for estimator_name, func in estimators.items()
            }
            for variable_name in samples.keys()
        }

        # remove unnecessary nesting
        estimates = {
            variable_name: {
                outer_key: squeeze_inner_estimates_dict(estimates[variable_name][outer_key])
                for outer_key in estimates[variable_name].keys()
            }
            for variable_name in estimates.keys()
        }

        return estimates

    def sample(
        self,
        *,
        num_samples: int,
        conditions: Mapping[str, np.ndarray],
        split: bool = False,
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
            Whether to split the output arrays along the last axis and return one column vector per target variable
            samples.
        **kwargs : dict
            Additional keyword arguments for the adapter and sampling process.

        Returns
        -------
        dict[str, np.ndarray]
            Dictionary containing generated samples with the same keys as `conditions`.
        """
        # Adapt, optionally standardize and convert conditions to tensor
        conditions = self._prepare_data(conditions, **kwargs)

        # Remove any superfluous keys, just retain actual conditions
        conditions = {k: v for k, v in conditions.items() if k in self.CONDITION_KEYS}

        # Sample and undo optional standardization
        samples = self._sample(num_samples=num_samples, **conditions, **kwargs)

        if "inference_variables" in self.standardize:
            samples = self.standardize_layers["inference_variables"](samples, forward=False)

        samples = {"inference_variables": samples}
        samples = keras.tree.map_structure(keras.ops.convert_to_numpy, samples)

        # Back-transform quantities and samples
        samples = self.adapter(samples, inverse=True, strict=False, **kwargs)

        if split:
            samples = split_arrays(samples, axis=-1)
        return samples

    def _prepare_data(
        self, data: Mapping[str, np.ndarray], log_det_jac: bool = False, **kwargs
    ) -> dict[str, Tensor] | tuple[dict[str, Tensor], dict[str, Tensor]]:
        """
        Adapts, (optionally) standardizes, and converts data to tensors for inference.

        Handles inputs containing only conditions, only inference_variables, or both.
        Optionally tracks log-determinant Jacobian (ldj) of transformations.
        """
        adapted = self.adapter(data, strict=False, stage="inference", log_det_jac=log_det_jac, **kwargs)

        if log_det_jac:
            data, ldj = adapted
            ldj_inference = ldj.get("inference_variables", 0.0)
        else:
            data = adapted
            ldj_inference = None

        # Standardize conditions
        for key in self.CONDITION_KEYS:
            if key in self.standardize and key in data:
                data[key] = self.standardize_layers[key](data[key])

        # Standardize inference variables
        if "inference_variables" in data and "inference_variables" in self.standardize:
            result = self.standardize_layers["inference_variables"](
                data["inference_variables"], log_det_jac=log_det_jac
            )
            if log_det_jac:
                data["inference_variables"], ldj_std = result
                ldj_inference += keras.ops.convert_to_numpy(ldj_std)
            else:
                data["inference_variables"] = result

        # Convert all data to tensors
        data = keras.tree.map_structure(keras.ops.convert_to_tensor, data)

        if log_det_jac:
            return data, ldj_inference
        return data

    def _sample(
        self,
        num_samples: int,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        **kwargs,
    ) -> Tensor:
        if self.summary_network is None:
            if summary_variables is not None:
                raise ValueError("Cannot use summary variables without a summary network.")
        else:
            if summary_variables is None:
                raise ValueError("Summary variables are required when a summary network is present.")

        if self.summary_network is not None:
            summary_outputs = self.summary_network(
                summary_variables, **filter_kwargs(kwargs, self.summary_network.call)
            )

            inference_conditions = concatenate_valid([inference_conditions, summary_outputs], axis=-1)

        if inference_conditions is not None:
            # conditions must always have shape (batch_size, ..., dims)
            batch_size = keras.ops.shape(inference_conditions)[0]
            inference_conditions = keras.ops.expand_dims(inference_conditions, axis=1)
            inference_conditions = keras.ops.broadcast_to(
                inference_conditions, (batch_size, num_samples, *keras.ops.shape(inference_conditions)[2:])
            )
            batch_shape = keras.ops.shape(inference_conditions)[:-1]
        else:
            batch_shape = keras.ops.shape(inference_conditions)[1:-1]

        return self.inference_network.sample(
            batch_shape, conditions=inference_conditions, **filter_kwargs(kwargs, self.inference_network.sample)
        )

    def summarize(self, data: Mapping[str, np.ndarray], **kwargs) -> np.ndarray:
        """
        Computes the learned summary statistics of given summary variables.

        The `data` dictionary is preprocessed using the `adapter` and passed through the summary network.

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
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

        data_adapted = self.adapter(data, strict=False, stage="inference", **kwargs)
        if "summary_variables" not in data_adapted or data_adapted["summary_variables"] is None:
            raise ValueError("Summary variables are required to compute summaries.")

        summary_variables = keras.tree.map_structure(keras.ops.convert_to_tensor, data_adapted["summary_variables"])
        summaries = self.summary_network(summary_variables, **filter_kwargs(kwargs, self.summary_network.call))
        summaries = keras.ops.convert_to_numpy(summaries)

        return summaries

    def summaries(self, data: Mapping[str, np.ndarray], **kwargs) -> np.ndarray:
        """
        .. deprecated:: 2.0.4
            `summaries` will be removed in version 2.0.6, it was renamed to `summarize` which should be used instead.
        """
        warnings.warn("`summaries` was renamed to `summarize` and will be removed in version 2.0.6.", FutureWarning)
        return self.summarize(data=data, **kwargs)

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
        # Adapt, optionally standardize and convert to tensor. Keep track of log_det_jac.
        data, log_det_jac = self._prepare_data(data, log_det_jac=True, **kwargs)

        # Pass data to networks and convert back to numpy array.
        log_prob = self._log_prob(**data, **kwargs)
        log_prob = keras.ops.convert_to_numpy(log_prob)

        # Change of variables formula.
        log_prob = log_prob + log_det_jac

        return log_prob

    def _log_prob(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        **kwargs,
    ) -> Tensor:
        if self.summary_network is None:
            if summary_variables is not None:
                raise ValueError("Cannot use summary variables without a summary network.")
        else:
            if summary_variables is None:
                raise ValueError("Summary variables are required when a summary network is present.")

        if self.summary_network is not None:
            summary_outputs = self.summary_network(
                summary_variables, **filter_kwargs(kwargs, self.summary_network.call)
            )
            inference_conditions = concatenate_valid((inference_conditions, summary_outputs), axis=-1)

        return self.inference_network.log_prob(
            inference_variables,
            conditions=inference_conditions,
            **filter_kwargs(kwargs, self.inference_network.log_prob),
        )

    def _batch_size_from_data(self, data: Mapping[str, any]) -> int:
        """
        Fetches the current batch size from an input dictionary. Can only be used during training when
        inference variables as present.
        """
        return keras.ops.shape(data["inference_variables"])[0]
