from collections.abc import Mapping, Sequence
from typing import Any

import multiprocessing as mp

import numpy as np
import keras

from bayesflow.adapters import Adapter
from bayesflow.datasets import OnlineDataset
from bayesflow.simulators import Simulator
from bayesflow.types import Tensor
from bayesflow.utils import find_batch_size, filter_kwargs, concatenate_valid_shapes, logging
from bayesflow.utils.serialization import deserialize, serialize

from .backend_approximators import BackendApproximator


class Approximator(BackendApproximator):
    """Base class for all BayesFlow approximators."""

    # Mask routing: {data_key → network_kwarg_name}.
    # Subclasses can narrow these to the masks they actually support.
    _SUMMARY_MASK_KEYS: dict[str, str] = {
        "summary_attention_mask": "attention_mask",
        "summary_mask": "mask",
    }
    _INFERENCE_MASK_KEYS: dict[str, str] = {
        "inference_attention_mask": "attention_mask",
        "inference_mask": "mask",
    }

    @staticmethod
    def _collect_mask_kwargs(mapping: dict[str, str], source: Mapping) -> dict:
        """Build a kwargs dict from a ``{source_key: target_kwarg_name}`` mapping.

        Looks up each *source_key* in *source*; when the value is not ``None``,
        it is added to the result under *target_kwarg_name*.
        """
        return {target: source[key] for key, target in mapping.items() if source.get(key) is not None}

    @property
    def standardize_layers(self):
        """Shortcut to the standardizer's per-variable layers."""

        if hasattr(self, "standardizer"):
            return self.standardizer.standardize_layers

    def build(self, data_shapes: Mapping[str, tuple[int] | Mapping[str, Mapping]]):
        """
        Template method for building all network components.

        This method orchestrates the build process by:
        1. Building the summary network (if present) and caching its output shape
        2. Enriching data_shapes with computed values for hooks to access
        3. Calling hook methods in the proper sequence
        4. Marking as built

        Hooks receive an enriched data_shapes dict that includes "_summary_outputs" if a
        summary network was built, so they don't need to recompute this value.
        """
        # Build standardizer first
        self._build_standardization_layers(data_shapes)

        # Build summary network once at template level and cache output shape
        summary_outputs_shape = self._build_summary_network(data_shapes)

        # Enrich data_shapes with computed summary output shape for hooks to access (if summary network was built)
        enriched_shapes = data_shapes.copy()
        if summary_outputs_shape is not None:
            enriched_shapes["_summary_outputs"] = summary_outputs_shape

        self._build_inference_network(enriched_shapes)

        self.built = True

    def _build_standardization_layers(self, data_shapes: Mapping[str, tuple[int] | Mapping]):
        """
        Helper method: builds the standardizer if present (default behavior for most approximators).
        """
        if hasattr(self, "standardizer") and not self.standardizer.built:
            self.standardizer.build(data_shapes)

    def _build_summary_network(self, data_shapes: Mapping[str, tuple[int] | Mapping]) -> tuple[int] | dict | None:
        """
        Helper method: builds the summary network if present.
        Subclasses can call this to build their summary network.

        Returns
        -------
        output_shape : tuple or dict or None
            The output shape of the summary network, or None if no summary network.
        """
        if not hasattr(self, "summary_network") or self.summary_network is None:
            return None

        if not self.summary_network.built:
            self.summary_network.build(data_shapes["summary_variables"])

        return self.summary_network.compute_output_shape(data_shapes["summary_variables"])

    def _build_inference_network(self, data_shapes: Mapping[str, tuple[int] | Mapping]):
        """
        Hook method: subclasses implement to build their inference network(s).
        Subclasses should call _build_summary_network() internally if needed.
        """

        if not self.inference_network.built:
            summary_outputs_shape = data_shapes.get("_summary_outputs")
            inference_conditions_shape = concatenate_valid_shapes(
                [data_shapes.get("inference_conditions"), summary_outputs_shape], axis=-1
            )
            self.inference_network.build(data_shapes["inference_variables"], inference_conditions_shape)

    def _prepare_conditions(
        self,
        data: Mapping[str, np.ndarray] | None,
        *,
        stage: str = "inference",
        summary_output: Tensor | np.ndarray | None = None,
        batch_size: int | None = None,
        **kwargs,
    ) -> tuple[dict[str, Tensor], Tensor | None, Tensor | None]:
        """Adapt raw user data, tensorize, standardize conditions, and resolve.

        Standard inference-time pipeline shared across all approximators:

        1. Apply the adapter (``strict=False``)
        2. Convert all values to tensors
        3. Standardize ``inference_conditions`` and ``summary_variables``
        4. Resolve conditions via the summary network (if present)

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            Raw user data dictionary.
        stage : str, optional
            Stage for standardization (default is ``"inference"``).
        summary_output: Tensor, optional
            Summarized data from a summary network, if already computed. If provided, this will be used instead of
            summarizing the raw data.
        batch_size : int, optional
            Batch size for the summary network (default is ``None``).
        **kwargs
            Extra keyword arguments forwarded to the adapter and summary network.

        Returns
        -------
        resolved_conditions : Tensor or None
            Standardized inference conditions concatenated with summary outputs.
        adapted : dict[str, Tensor] or {}
            The full adapted and tensorized dictionary or {} if data is None or empty.
        summary_outputs : Tensor or None
            Raw summary network outputs, or ``None`` if no summary network.
        """

        if not data and not summary_output:
            return None, {}, None

        adapted = self.adapter(data, strict=False, **kwargs)
        adapted = keras.tree.map_structure(keras.ops.convert_to_tensor, adapted)

        summary_kwargs = self._collect_mask_kwargs(self._SUMMARY_MASK_KEYS, adapted)

        resolved_conditions, summary_outputs = self._standardize_and_resolve(
            inference_conditions=adapted.get("inference_conditions"),
            summary_variables=adapted.get("summary_variables"),
            summary_output=summary_output,
            stage=stage,
            batch_size=batch_size,
            **summary_kwargs,
        )
        return resolved_conditions, adapted, summary_outputs

    def _prepare_compositional_conditions(
        self,
        conditions: Mapping[str, np.ndarray],
        batch_size: int | None = None,
        summary_output: Tensor | np.ndarray | None = None,
        **kwargs,
    ) -> tuple[Tensor | None, dict[str, Tensor], Tensor | None]:
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

        if summary_output is not None:
            summary_output = keras.ops.reshape(
                summary_output, (n_datasets * n_comp,) + keras.ops.shape(summary_output)[1:]
            )

        resolved_conditions, adapted, summary_outputs = self._prepare_conditions(
            data=flattened_conditions,
            summary_output=summary_output,
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
        summary_outputs = keras.ops.reshape(
            summary_outputs,
            (n_datasets, n_comp) + keras.ops.shape(summary_outputs)[1:],
        )
        return resolved_conditions, adapted, summary_outputs

    def _prepare_ancestral_conditions(
        self,
        conditions: Mapping[str, np.ndarray],
        ancestral_conditions: Mapping[str, np.ndarray],
        batch_size: int | None = None,
        **adapter_kwargs,
    ) -> tuple[Tensor | None, dict[str, Tensor], Tensor | None]:
        first_conditions_arr = np.asarray(next(iter(conditions.values())))
        first_ancestral_arr = np.asarray(next(iter(ancestral_conditions.values())))

        n_datasets = first_conditions_arr.shape[0]
        n_children = first_conditions_arr.shape[1]
        n_parent_samples = first_ancestral_arr.shape[1]

        # If a summary network is present, summarize child conditions before expansion
        summary_network = getattr(self, "summary_network", None)
        if summary_network is not None:
            flat_child_batch = n_datasets * n_children
            flattened_conditions = {
                key: np.asarray(value).reshape(flat_child_batch, *np.asarray(value).shape[2:])
                for key, value in conditions.items()
            }
            adapted_child = self.adapter(flattened_conditions, strict=False, **adapter_kwargs)
            adapted_child = keras.tree.map_structure(keras.ops.convert_to_tensor, adapted_child)

            summary_kwargs = self._collect_mask_kwargs(self._SUMMARY_MASK_KEYS, adapted_child)
            child_summary_variables = self.standardizer.maybe_standardize(
                adapted_child.get("summary_variables"), key="summary_variables", stage="inference"
            )
        else:
            summary_kwargs = {}
            child_summary_variables = None

        flat_batch = n_datasets * n_children * n_parent_samples
        # (n_datasets, n_parent_samples, ...) -> (n_datasets, n_children, n_parent_samples, ...) -> (flat_batch, ...)
        expanded_ancestral = {}
        for key, value in ancestral_conditions.items():
            arr = np.asarray(value)
            arr = np.expand_dims(arr, axis=1)  # (n_datasets, 1, n_parent_samples, ...)
            arr = np.repeat(arr, n_children, axis=1)  # (n_datasets, n_children, n_parent_samples, ...)
            expanded_ancestral[key] = arr.reshape(flat_batch, *arr.shape[3:])

        # (n_datasets, n_children, ...) -> (n_datasets, n_children, n_parent_samples, ...) -> (flat_batch, ...)
        expanded_conditions = {}
        for key, value in conditions.items():
            arr = np.asarray(value)
            arr = np.expand_dims(arr, axis=2)  # (n_datasets, n_children, 1, ...)
            arr = np.repeat(arr, n_parent_samples, axis=2)  # (n_datasets, n_children, n_parent_samples, ...)
            expanded_conditions[key] = arr.reshape(flat_batch, *arr.shape[3:])

        merged_conditions = {**expanded_conditions, **expanded_ancestral}
        adapted = self.adapter(merged_conditions, strict=False, **adapter_kwargs)
        adapted = keras.tree.map_structure(keras.ops.convert_to_tensor, adapted)

        inference_conditions = self.standardizer.maybe_standardize(
            adapted.get("inference_conditions"), key="inference_conditions", stage="inference"
        )

        resolved_conditions, summary_outputs = self.condition_builder.resolve_ancestral(
            summary_network,
            inference_conditions,
            child_summary_variables,
            n_datasets=n_datasets,
            n_children=n_children,
            n_parent_samples=n_parent_samples,
            batch_size=batch_size,
            **summary_kwargs,
        )

        return resolved_conditions, adapted, summary_outputs

    def _standardize_and_resolve(
        self,
        inference_conditions: Tensor | None,
        summary_variables: Tensor | None,
        *,
        stage: str,
        summary_output: Tensor | np.ndarray | None = None,
        batch_size: int | None = None,
        purpose: str = "call",
        **summary_kwargs,
    ):
        """Standardize condition tensors and resolve via the summary network.

        Shared by both inference-time methods (via :meth:`_prepare_conditions`)
        and training-time ``compute_metrics`` implementations.

        Parameters
        ----------
        inference_conditions : Tensor or None
            Inference conditions (pre-adapted tensors).
        summary_variables : Tensor or None
            Summary variables (pre-adapted tensors).
        stage : str
            Current stage (``"training"``, ``"validation"``, or ``"inference"``).
        summary_output : Tensor or None
            If already computed, the output of the summary network. If provided, this will be used instead of
            computing summaries again from summary variables.
        batch_size : int, optional
            Batch size for the summary network (default is ``None``).
        purpose : str, optional
            Passed to :meth:`ConditionBuilder.resolve` — ``"call"`` for forward
            passes, ``"metrics"`` for training/validation (default is ``"call"``).
        **summary_kwargs
            Extra keyword arguments forwarded to
            :meth:`ConditionBuilder.resolve` and ultimately to the summary
            network's ``call`` / ``compute_metrics`` method (e.g.
            ``attention_mask``).

        Returns
        -------
        resolved_conditions : Tensor or None
            Standardized inference conditions concatenated with summary outputs.
        summary_output : Tensor, dict, or None
            For ``purpose="call"``: summary network output tensor or ``None``.
            For ``purpose="metrics"``: dict of summary metrics.
        """
        inference_conditions = self.standardizer.maybe_standardize(
            inference_conditions, key="inference_conditions", stage=stage
        )
        summary_variables = self.standardizer.maybe_standardize(summary_variables, key="summary_variables", stage=stage)
        resolved_conditions, summary_output = self.condition_builder.resolve(
            summary_network=self.summary_network,
            inference_conditions=inference_conditions,
            summary_variables=summary_variables,
            summary_output=summary_output,
            stage=stage,
            batch_size=batch_size,
            purpose=purpose,
            **summary_kwargs,
        )
        return resolved_conditions, summary_output

    @classmethod
    def build_adapter(
        cls,
        inference_variables: str | Sequence[str],
        inference_conditions: str | Sequence[str] = None,
        summary_variables: str | Sequence[str] = None,
        sample_weight: str = None,
        summary_attention_mask: str = None,
        summary_mask: str = None,
        inference_attention_mask: str = None,
        inference_mask: str = None,
    ) -> Adapter:
        """Create a default :py:class:`~bayesflow.adapters.Adapter` for the approximator.

        Handles the common pipeline shared by all approximators:
        ``to_array -> convert_dtype -> concatenate -> keep``.
        Subclasses can call ``super().build_adapter(...)`` and apply additional
        steps to the returned adapter.

        Parameters
        ----------
        inference_variables : str or Sequence[str]
            Names of the inference variables in the data dict.
        inference_conditions : str or Sequence[str], optional
            Names of the inference conditions in the data dict.
        summary_variables : str or Sequence[str], optional
            Names of the summary variables in the data dict.
        sample_weight : str, optional
            Name of the sample weight variable.
        summary_attention_mask : str, optional
            Name of the attention mask for the summary network.
            Forwarded as ``attention_mask`` to the summary network.
        summary_mask : str, optional
            Name of the padding/key mask for the summary network.
            Forwarded as ``mask`` to the summary network.
        inference_attention_mask : str, optional
            Name of the attention mask for the inference network.
            Forwarded as ``attention_mask`` to the inference network.
        inference_mask : str, optional
            Name of the padding/key mask for the inference network.
            Forwarded as ``mask`` to the inference network.
        """

        if isinstance(inference_variables, str):
            inference_variables = [inference_variables]
        if isinstance(inference_conditions, str):
            inference_conditions = [inference_conditions]
        if isinstance(summary_variables, str):
            summary_variables = [summary_variables]

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
            adapter.rename(sample_weight, "sample_weight")

        keep = ["inference_variables", "inference_conditions", "summary_variables", "sample_weight"]

        for canonical_name, user_name in {
            "summary_attention_mask": summary_attention_mask,
            "summary_mask": summary_mask,
            "inference_attention_mask": inference_attention_mask,
            "inference_mask": inference_mask,
        }.items():
            if user_name is not None:
                adapter.rename(user_name, canonical_name)
                keep.append(canonical_name)

        adapter.keep(keep)

        return adapter

    def build_dataset(
        self,
        *,
        batch_size: int = "auto",
        num_batches: int,
        adapter: Adapter = "auto",
        memory_budget: str | int = "auto",
        simulator: Simulator,
        workers: int = "auto",
        use_multiprocessing: bool = False,
        max_queue_size: int = 32,
        **kwargs,
    ) -> OnlineDataset:
        if batch_size == "auto":
            batch_size = find_batch_size(memory_budget=memory_budget, sample=simulator.sample(1))
            logging.info(f"Using a batch size of {batch_size}.")

        if adapter == "auto":
            adapter = self.build_adapter(**filter_kwargs(kwargs, self.build_adapter))

        if workers == "auto":
            workers = mp.cpu_count()
            logging.info(f"Using {workers} data loading workers.")

        workers = workers or 1

        return OnlineDataset(
            simulator=simulator,
            batch_size=batch_size,
            num_batches=num_batches,
            adapter=adapter,
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size,
        )

    def call(self, *args, **kwargs):
        return self.compute_metrics(*args, **kwargs)

    def compile(
        self,
        *args,
        inference_metrics: Any = None,
        summary_metrics: Any = None,
        **kwargs,
    ):
        """
        Compile the approximator, setting metrics on inference and summary networks if provided.

        Parameters
        ----------
        inference_metrics : keras.Metric or Sequence[keras.Metric], optional
            Metric(s) to set on the inference_network.
        summary_metrics : keras.Metric or Sequence[keras.Metric], optional
            Metric(s) to set on the summary_network (if present).
        *args, **kwargs
            Additional arguments passed to the parent compile method.
        """
        if inference_metrics:
            self.inference_network._metrics = inference_metrics

        if summary_metrics:
            if not hasattr(self, "summary_network") or self.summary_network is None:
                logging.warning("Ignoring summary metrics because there is no summary network.")
            else:
                self.summary_network._metrics = summary_metrics

        return super().compile(*args, **kwargs)

    def fit(self, *, dataset: keras.utils.PyDataset = None, simulator: Simulator = None, **kwargs):
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

        if dataset is None:
            if simulator is None:
                raise ValueError("Received no data to fit on. Please provide either a dataset or a simulator.")

            logging.info(f"Building dataset from simulator instance of {simulator.__class__.__name__}.")
            dataset = self.build_dataset(simulator=simulator, **filter_kwargs(kwargs, self.build_dataset))
        else:
            if simulator is not None:
                raise ValueError(
                    "Received conflicting arguments. Please provide either a dataset or a simulator, but not both."
                )

            logging.info(f"Fitting on dataset instance of {dataset.__class__.__name__}.")

        if not self.built:
            logging.info("Building on a test batch.")
            mock_data = dataset[0]
            mock_data = keras.tree.map_structure(keras.ops.convert_to_tensor, mock_data)
            self.build_from_data(mock_data)

        return super().fit(dataset=dataset, **kwargs)

    def build_from_data(self, adapted_data: Mapping[str, Any]):
        """Build the approximator from adapted data by extracting shapes."""
        self.build(keras.tree.map_structure(keras.ops.shape, adapted_data))

    def compile_from_config(self, config):
        """Compile the approximator from a saved configuration."""
        self.compile(**deserialize(config))
        if hasattr(self, "optimizer") and self.built:
            self.optimizer.build(self.trainable_variables)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        """Deserialize and instantiate an approximator from configuration."""
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_compile_config(self):
        """
        Serialize compile configuration for all network metrics.

        Collects metrics from inference_network and summary_network (if present),
        serializes them, and merges with parent class config.

        Returns
        -------
        dict
            Configuration dictionary with serialized metrics.
        """
        base_config = super().get_compile_config() or {}

        config = {}

        if hasattr(self, "inference_network") and self.inference_network is not None:
            config["inference_metrics"] = self.inference_network._metrics

        if hasattr(self, "summary_network") and self.summary_network is not None:
            config["summary_metrics"] = self.summary_network._metrics

        return base_config | serialize(config)

    def summarize(self, conditions: Mapping[str, np.ndarray], **kwargs) -> np.ndarray:
        """
        Computes the learned summary statistics of given summary variables.

        The `conditions` dictionary is preprocessed using the `adapter` and passed through the summary network.

        Parameters
        ----------
        conditions : Mapping[str, np.ndarray]
            Dictionary of simulated or real quantities as NumPy arrays.
        **kwargs : dict
            Additional keyword arguments for the adapter and the summary network.

        Returns
        -------
        summaries : np.ndarray
            The learned summary statistics. Returns None if no summary network is present.
        """
        if not conditions:
            raise ValueError("No valid conditions provided for summarization.")

        if not hasattr(self, "summary_network") or self.summary_network is None:
            raise ValueError("Summary network is not available. This approximator does not support summarization.")

        if not hasattr(self, "adapter"):
            raise ValueError("Adapter is not available.")

        _, _, summary_outputs = self._prepare_conditions(conditions, **kwargs)

        return keras.ops.convert_to_numpy(summary_outputs)

    def _batch_size_from_data(self, data: Mapping[str, any]) -> int:
        """Return the batch size from a training data dict.

        Relies on the ``"inference_variables"`` key, which is present in
        every approximator's training data.
        """
        return keras.ops.shape(data["inference_variables"])[0]
