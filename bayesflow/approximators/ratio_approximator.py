from collections.abc import Sequence, Mapping

import numpy as np
import keras

from bayesflow.types import Tensor
from bayesflow.adapters import Adapter
from bayesflow.utils import expand_tile, concatenate_valid_shapes
from bayesflow.utils.serialization import serialize, serializable

from .approximator import Approximator
from .helpers import ConditionBuilder

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
    inference_network : keras.Layer
        A network backbone to perform contrastive learning. Last logits layer
        is automatically added on top of the inference network.
    summary_network : SummaryNetwork, optional
        The summary network used for data summarization of summary_variables (default is None).
        When present, summary outputs are automatically concatenated with inference_conditions.
    gamma: float, optional
        Odds or of any pair being drawn dependently to completely independently.
        Default is 1.
    K: int, optional
        Number of parameter candidates used for contrastive learning.
        Default is 5.
    standardize : str | Sequence[str] | None
        The variables to standardize before passing to the networks. Can be either
        "all" or any subset of ["inference_variables", "inference_conditions", "summary_variables"].
        (default is "inference_variables").
    **kwargs : dict, optional
        Additional arguments passed to the :py:class:`bayesflow.approximators.Approximator` class.
    """

    def __init__(
        self,
        adapter: Adapter,
        inference_network: keras.Layer,
        summary_network: keras.Layer = None,
        gamma: float = 1.0,
        K: int = 5,
        standardize: str | Sequence[str] | None = "inference_variables",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.adapter = adapter
        self.inference_network = inference_network
        self.summary_network = summary_network
        self.condition_builder = ConditionBuilder()

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

        self.standardizer = Standardization(standardize)

    def build(self, data_shapes: Mapping[str, Tensor]):
        self._build_standardization_layers(data_shapes)

        # Build summary network once at template level and cache output shape
        summary_outputs_shape = self._build_summary_network(data_shapes)

        classifier_inputs_shape = concatenate_valid_shapes(
            [data_shapes["inference_variables"], data_shapes["inference_conditions"], summary_outputs_shape], axis=-1
        )

        if not self.inference_network.built:
            self.inference_network.build(classifier_inputs_shape)

        classifier_outputs_shape = self.inference_network.compute_output_shape(classifier_inputs_shape)

        if not self.projector.built:
            self.projector.build(classifier_outputs_shape)

        self.built = True

    def compute_metrics(
        self,
        inference_variables: Tensor,
        inference_conditions: Tensor = None,
        summary_variables: Tensor = None,
        stage: str = "training",
    ) -> dict[str, Tensor]:
        """
        Computes loss following https://arxiv.org/pdf/2210.06170.

        Handles both summary network outputs (if present) and inference conditions,
        combining them via ConditionBuilder.resolve().
        """
        inference_variables = self.standardizer.maybe_standardize(
            inference_variables, key="inference_variables", stage=stage
        )

        resolved_conditions, summary_metrics = self._standardize_and_resolve(
            inference_conditions, summary_variables, stage=stage, purpose="metrics"
        )

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

        # Get (batch_size, K, dim) conditions (already resolved from condition builder)
        conditions = expand_tile(resolved_conditions, n=self.K, axis=1)

        marginal_logits = self.logits(bootstrap_inference_variables[:, 1:, :], conditions, stage=stage)
        joint_logits = self.logits(bootstrap_inference_variables[:, :-1, :], conditions, stage=stage)

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
        inference_loss = keras.ops.mean(loss)

        # Handle summary network metrics if present
        if "loss" in summary_metrics:
            total_loss = inference_loss + summary_metrics["loss"]
        else:
            total_loss = inference_loss

        # Format metrics with prefixes
        inference_metrics = {"loss": total_loss}
        summary_metrics = {f"{key}/summary_{key}": value for key, value in summary_metrics.items()}

        metrics = inference_metrics | summary_metrics
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

    def log_ratio(self, data: Mapping[str, np.ndarray], **kwargs) -> Tensor:
        """
        Computes the log likelihood-to-evidence ratio.
        The `data` dictionary is preprocessed using the `adapter`.

        Parameters
        ----------
        data : Mapping[str, np.ndarray]
            Dictionary of observed data as NumPy arrays.
        **kwargs : dict
            Additional keyword arguments for the adapter and log-probability computation.

        Returns
        -------
        log_ratio: Tensor
            The estimated log ratios.
        """

        resolved_conditions, adapted, _ = self._prepare_conditions(data, **kwargs)

        inference_variables = self.standardizer.maybe_standardize(
            adapted.get("inference_variables"), key="inference_variables", stage="inference"
        )

        log_ratio = self.logits(inference_variables, resolved_conditions, stage="inference")
        return log_ratio

    def logits(self, inference_variables: Tensor, inference_conditions: Tensor, stage: str) -> Tensor:
        """Computes logits for K batches of variables-conditions pairs."""
        classifier_inputs = keras.ops.concatenate([inference_variables, inference_conditions], axis=-1)
        logits = self.inference_network(classifier_inputs, training=stage == "training")
        logits = self.projector(logits)
        logits = keras.ops.squeeze(logits, axis=-1)
        return logits

    def get_config(self):
        base_config = super().get_config()
        config = {
            "adapter": self.adapter,
            "summary_network": self.summary_network,
            "inference_network": self.inference_network,
            "gamma": self.gamma,
            "K": self.K,
            "standardize": self.standardizer.standardize,
        }

        return base_config | serialize(config)

    def _sample_from_batch(self, inference_variables: Tensor) -> Tensor:
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
