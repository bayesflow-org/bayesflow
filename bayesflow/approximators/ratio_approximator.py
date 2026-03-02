from collections.abc import Sequence

import keras

from bayesflow.types import Tensor
from bayesflow.adapters import Adapter
from bayesflow.utils import expand_tile, concatenate_valid_shapes
from bayesflow.utils.serialization import serialize, deserialize, serializable

from .approximator import Approximator
from ..networks.standardization import Standardization


@serializable("bayesflow.approximators")
class RatioApproximator(Approximator):
    """
    Implements contrastive neural likelihood-to-evidence ratio estimation  (NRE-C)
    as described in https://arxiv.org/pdf/2210.06170.
    The estimation target is the ratio of likelihood and evidence: p(x | theta) / p(x).

    Parameters
    ----------
    adapter : bayesflow.adapters.Adapter
        Adapter for data processing. You can use :py:meth:`build_adapter`
        to create it.
    classifier_network : A classification network to perform contrastive learning.
    summary_network : SummaryNetwork, optional
        The summary network used for data summarization (default is None).
    gamma: float, optional
        Odds or of any pair being drawn dependently to completely independently.
        Default is 1.
    K: int, optional
        Number of parameter candidates used for contrastive learning.
        Default is 5.
    **kwargs : dict, optional
        Additional arguments passed to the :py:class:`bayesflow.approximators.Approximator` class.
    """

    def __init__(
        self,
        adapter: Adapter,
        classifier_network: keras.Layer,
        summary_network: keras.Layer = None,
        gamma: float = 1.0,
        K: int = 5,
        standardize: str | Sequence[str] | None = "all",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.adapter = adapter
        self.classifier_network = classifier_network
        self.summary_network = summary_network

        if gamma <= 0:
            raise ValueError(f"Gamma must be positive, got {gamma}.")
        if gamma == float("inf"):
            raise NotImplementedError("NRE-B is not yet supported.")
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}.")

        self.gamma = gamma
        self.K = K

        self.projector = keras.layers.Dense(units=1)
        self.seed_generator = keras.random.SeedGenerator()

        # standardization has to come last in the constructor
        if isinstance(standardize, str) and standardize != "all":
            self.standardize = [standardize]
        else:
            self.standardize = standardize or []

        if self.standardize == "all":
            # we have to lazily initialize these
            self.standardize_layers = None
        else:
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

    def build(self, data_shapes):
        if self.summary_network is not None:
            if not self.summary_network.built:
                self.summary_network.build(data_shapes["inference_conditions"])
            inference_conditions_shape = self.summary_network.compute_output_shape(data_shapes["inference_conditions"])
        else:
            inference_conditions_shape = data_shapes["inference_conditions"]

        # Compute inference_conditions_shape by combining original and summary outputs
        classifier_inputs_shape = concatenate_valid_shapes(
            [data_shapes["inference_variables"], inference_conditions_shape], axis=-1
        )

        # Build inference network if needed
        if not self.classifier_network.built:
            self.classifier_network.build(classifier_inputs_shape)
        classifier_outputs_shape = self.classifier_network.compute_output_shape(classifier_inputs_shape)

        if not self.projector.built:
            self.projector.build(classifier_outputs_shape)

        # Set up standardization layers if requested
        if self.standardize == "all":
            # Only include variables present in data_shapes
            self.standardize = [var for var in ["inference_variables", "inference_conditions"] if var in data_shapes]
            self.standardize_layers = {var: Standardization(trainable=False) for var in self.standardize}

        # Build all standardization layers
        for var, layer in self.standardize_layers.items():
            layer.build(data_shapes[var])

        self.built = True

    def build_from_data(self, adapted_data: dict[str, any]) -> None:
        data_shapes = keras.tree.map_structure(keras.ops.shape, adapted_data)
        self.build(data_shapes)

    def compute_metrics(self, inference_variables: Tensor, inference_conditions: Tensor, stage: str = "training"):
        """Computes loss following https://arxiv.org/pdf/2210.06170"""

        if "inference_variables" in self.standardize:
            inference_variables = self.standardize_layers["inference_variables"](inference_variables, stage=stage)

        if "inference_conditions" in self.standardize:
            inference_conditions = self.standardize_layers["inference_conditions"](inference_conditions, stage=stage)

        batch_size = keras.ops.shape(inference_variables)[0]

        log_gamma = keras.ops.broadcast_to(keras.ops.log(self.gamma), (batch_size,))
        log_K = keras.ops.broadcast_to(keras.ops.log(self.K), (batch_size,))

        marginal_weight = 1 / (1 + self.gamma)
        joint_weight = self.gamma / (1 + self.gamma)

        # Get (batch_size, K+1, dim) inference variables (theta)
        bootstrap_inference_variables = self._sample_from_batch(inference_variables)
        bootstrap_inference_variables = keras.ops.concatenate(
            [inference_variables[:, None, :], bootstrap_inference_variables], axis=1
        )

        # Get (batch_size, K, dim) conditions
        if self.summary_network is not None:
            inference_conditions = self.summary_network(inference_conditions, training=stage == "training")
        inference_conditions = expand_tile(inference_conditions, n=self.K, axis=1)

        marginal_logits = self.logits(bootstrap_inference_variables[:, 1:, :], inference_conditions, stage=stage)
        joint_logits = self.logits(bootstrap_inference_variables[:, :-1, :], inference_conditions, stage=stage)

        # Eq. 7 (https://arxiv.org/abs/2210.06170) - we use a trick for numerical stability:
        # log(K + gamma * sum_{i=1}^{K} exp(h_i)) = log(exp(log K) + sum_{i=1}^{K} exp(h_i + log gamma))
        # so if we absorb log gamma into the network outputs and concatenate log K, we can use logsumexp

        log_numerator_joint = log_gamma + joint_logits[:, 0]
        log_denominator_joint = keras.ops.concatenate([log_gamma[:, None] + joint_logits, log_K[:, None]], axis=-1)
        log_denominator_joint = keras.ops.logsumexp(log_denominator_joint, axis=-1)

        log_numerator_marginal = log_K
        log_denominator_marginal = keras.ops.concatenate(
            [log_gamma[:, None] + marginal_logits, log_K[:, None]], axis=-1
        )
        log_denominator_marginal = keras.ops.logsumexp(log_denominator_marginal, axis=-1)

        joint_loss = log_denominator_joint - log_numerator_joint
        marginal_loss = log_denominator_marginal - log_numerator_marginal

        loss = marginal_weight * marginal_loss + joint_weight * joint_loss
        loss = keras.ops.mean(loss)

        return {"loss": loss}

    def _sample_from_batch(self, inference_variables: Tensor, seed=None):
        """Samples K batches of inference variables with replacement. Ensures
        that no self-sampling occurs (i.e., all samples are negative examples)."""
        B = keras.ops.shape(inference_variables)[0]

        r = keras.random.randint(
            shape=(B, self.K),
            minval=0,
            maxval=B - 1,
            dtype="int32",
            seed=self.seed_generator,
        )

        i = keras.ops.expand_dims(keras.ops.arange(B, dtype="int32"), axis=1)
        idx = r + keras.ops.cast(r >= i, "int32")

        return keras.ops.take(inference_variables, idx, axis=0)

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

    def log_ratio(self, data: dict[str, Tensor], **kwargs):
        """
        Computes the log likelihood-to-evidence ratio.
        The `data` dictionary is preprocessed using the `adapter`.
        Log-ratios are returned as NumPy arrays.

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            Dictionary of observed data as NumPy arrays.
        **kwargs : dict
            Additional keyword arguments for the adapter and log-probability computation.

        Returns
        -------
        np.ndarray
        """
        adapted = self.adapter(data, strict=False, **kwargs)
        inference_variables = adapted["inference_variables"]
        inference_conditions = adapted.get("inference_conditions")

        if "inference_variables" in self.standardize:
            inference_variables = self.standardize_layers["inference_variables"](inference_variables)

        if "inference_conditions" in self.standardize:
            inference_conditions = self.standardize_layers["inference_conditions"](inference_conditions)

        if self.summary_network is not None:
            inference_conditions = self.summary_network(inference_conditions, training=False)

        log_ratio = self.logits(inference_variables, inference_conditions, stage="inference")
        return log_ratio

    def logits(self, inference_variables: Tensor, inference_conditions: Tensor, stage: str):
        """Computes logits for K batches of variables-conditions pairs."""
        classifier_inputs = keras.ops.concatenate([inference_variables, inference_conditions], axis=-1)
        logits = self.classifier_network(classifier_inputs, training=stage == "training")
        logits = self.projector(logits)
        logits = keras.ops.squeeze(logits, axis=-1)
        return logits

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def get_config(self):
        base_config = super().get_config()
        config = {
            "adapter": self.adapter,
            "summary_network": self.summary_network,
            "classifier_network": self.classifier_network,
            "gamma": self.gamma,
            "K": self.K,
            "standardize": self.standardize,
        }

        return base_config | serialize(config)

    @classmethod
    def build_adapter(
        cls, inference_variables: str | Sequence[str], inference_conditions: str | Sequence[str]
    ) -> Adapter:
        """Produce a default adapter that converts dtypes and renames variables."""

        if isinstance(inference_variables, str):
            inference_variables = [inference_variables]
        if isinstance(inference_conditions, str):
            inference_conditions = [inference_conditions]

        adapter = Adapter()
        adapter.to_array()
        adapter.convert_dtype("float64", "float32")
        adapter.concatenate(inference_variables, into="inference_variables")
        adapter.concatenate(inference_conditions, into="inference_conditions")
        return adapter

    def _batch_size_from_data(self, data: any) -> int:
        return keras.ops.shape(data["inference_variables"])[0]
