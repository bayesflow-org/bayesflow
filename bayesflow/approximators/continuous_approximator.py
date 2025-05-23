from collections.abc import Mapping, Sequence, Callable

import numpy as np

import keras

from bayesflow.adapters import Adapter
from bayesflow.networks import InferenceNetwork, SummaryNetwork
from bayesflow.types import Tensor
from bayesflow.utils import filter_kwargs, logging, split_arrays, squeeze_inner_estimates_dict
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
    **kwargs : dict, optional
        Additional arguments passed to the :py:class:`bayesflow.approximators.Approximator` class.
    """

    SAMPLE_KEYS = ["summary_variables", "inference_conditions"]

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
        self.standardize = standardize

        self.inference_variables_norm = None
        self.summary_variables_norm = None
        self.inference_conditions_norm = None

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
            Names of the inference variables in the data
        inference_conditions : Sequence of str, optional
            Names of the inference conditions in the data
        summary_variables : Sequence of str, optional
            Names of the summary variables in the data
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
        # Determine input standardization
        if self.standardize == "all":
            keys = ["inference_variables", "summary_variables", "inference_conditions"]
        elif isinstance(self.standardize, str):
            keys = [self.standardize]
        elif isinstance(self.standardize, Sequence):
            keys = self.standardize
        else:
            keys = []

        if "inference_variables" in adapted_data and "inference_variables" in keys:
            self.inference_variables_norm = Standardization()

        if "summary_variables" in adapted_data and "summary_variables" in keys and self.summary_network:
            self.summary_variables_norm = Standardization()

        if "inference_conditions" in adapted_data and "inference_conditions" in keys:
            self.inference_conditions_norm = Standardization()

        # Call compute metrics once to build inner networks
        self.compute_metrics(**filter_kwargs(adapted_data, self.compute_metrics), stage="training")
        self.built = True

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
        # Optionally standardize optional inference conditions
        if inference_conditions is not None and self.inference_conditions_norm:
            inference_conditions = self.inference_conditions_norm(inference_conditions, stage=stage)

        if self.summary_network is None:
            if summary_variables is not None:
                raise ValueError("Cannot compute summary metrics without a summary network.")

            summary_metrics = {}
        else:
            if summary_variables is None:
                raise ValueError("Summary variables are required when a summary network is present.")

            if self.summary_variables_norm is not None:
                summary_variables = self.summary_variables_norm(summary_variables, stage=stage)

            summary_metrics = self.summary_network.compute_metrics(summary_variables, stage=stage)
            summary_outputs = summary_metrics.pop("outputs")

            # append summary outputs to inference conditions
            if inference_conditions is None:
                inference_conditions = summary_outputs
            else:
                inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=-1)

        # Force a conversion to Tensor
        inference_variables = keras.tree.map_structure(keras.ops.convert_to_tensor, inference_variables)

        if self.inference_variables_norm is not None:
            inference_variables = self.inference_variables_norm(inference_variables, stage=stage)

        inference_metrics = self.inference_network.compute_metrics(
            inference_variables, conditions=inference_conditions, sample_weight=sample_weight, stage=stage
        )

        loss = inference_metrics.get("loss", keras.ops.zeros(())) + summary_metrics.get("loss", keras.ops.zeros(()))

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

        # Apply adapter transforms to raw simulated / real quantities
        conditions = self.adapter(conditions, strict=False, stage="inference", **kwargs)

        # Ensure only keys relevant for sampling are present in the conditions dictionary
        conditions = {k: v for k, v in conditions.items() if k in ContinuousApproximator.SAMPLE_KEYS}

        # Optionally standardize conditions
        if "summary_variables" in conditions and self.summary_variables_norm:
            conditions["summary_variables"] = self.summary_variables_norm(conditions["summary_variables"])

        if "inference_conditions" in conditions and self.inference_conditions_norm:
            conditions["inference_conditions"] = self.inference_conditions_norm(conditions["inference_conditions"])

        conditions = keras.tree.map_structure(keras.ops.convert_to_tensor, conditions)

        # Sample and undo optional standardization
        samples = self._sample(num_samples=num_samples, **conditions, **kwargs)

        if self.inference_variables_norm:
            samples = self.inference_variables_norm(samples, forward=False)

        samples = {"inference_variables": samples}
        samples = keras.tree.map_structure(keras.ops.convert_to_numpy, samples)

        # Back-transform quantities and samples
        samples = self.adapter(samples, inverse=True, strict=False, **kwargs)

        if split:
            samples = split_arrays(samples, axis=-1)
        return samples

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

            summary_outputs = self.summary_network(
                summary_variables, **filter_kwargs(kwargs, self.summary_network.call)
            )

            if inference_conditions is None:
                inference_conditions = summary_outputs
            else:
                inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=1)

        if inference_conditions is not None:
            # conditions must always have shape (batch_size, dims)
            batch_size = keras.ops.shape(inference_conditions)[0]
            inference_conditions = keras.ops.expand_dims(inference_conditions, axis=1)
            inference_conditions = keras.ops.broadcast_to(
                inference_conditions, (batch_size, num_samples, *keras.ops.shape(inference_conditions)[2:])
            )
            batch_shape = (batch_size, num_samples)
        else:
            batch_shape = (num_samples,)

        return self.inference_network.sample(
            batch_shape,
            conditions=inference_conditions,
            **filter_kwargs(kwargs, self.inference_network.sample),
        )

    def summaries(self, data: Mapping[str, np.ndarray], **kwargs) -> np.ndarray:
        """
        Computes the learned summary statistics of given inputs.

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

        summary_variables = keras.ops.convert_to_tensor(data_adapted["summary_variables"])
        summaries = self.summary_network(summary_variables, **filter_kwargs(kwargs, self.summary_network.call))
        summaries = keras.ops.convert_to_numpy(summaries)

        return summaries

    def log_prob(self, data: Mapping[str, np.ndarray], **kwargs) -> np.ndarray | dict[str, np.ndarray]:
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
        data, log_det_jac = self.adapter(data, strict=False, stage="inference", log_det_jac=True, **kwargs)

        # Optionally standardize conditions and variables
        if "summary_variables" in data and self.summary_variables_norm:
            data["summary_variables"] = self.summary_variables_norm(data["summary_variables"])

        if "inference_conditions" in data and self.inference_conditions_norm:
            data["inference_conditions"] = self.inference_conditions_norm(data["inference_conditions"])

        if self.inference_variables_norm:
            data["inference_variables"], log_det_jac = self.summary_variables_norm(
                data["inference_variables"], log_det_jac=True
            )
            log_det_jac = keras.ops.convert_to_numpy(log_det_jac)
        else:
            log_det_jac = 0.0

        data = keras.tree.map_structure(keras.ops.convert_to_tensor, data)
        log_prob = self._log_prob(**data, **kwargs)
        log_prob = keras.tree.map_structure(keras.ops.convert_to_numpy, log_prob)

        # change of variables formula
        log_det_jac = log_det_jac.get("inference_variables")
        if log_det_jac is not None:
            log_prob = keras.tree.map_structure(lambda x: x + log_det_jac, log_prob)

        return log_prob

    def _log_prob(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        **kwargs,
    ) -> Tensor | dict[str, Tensor]:
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
                inference_conditions = keras.ops.concatenate([inference_conditions, summary_outputs], axis=-1)

        return self.inference_network.log_prob(
            inference_variables,
            conditions=inference_conditions,
            **filter_kwargs(kwargs, self.inference_network.log_prob),
        )
