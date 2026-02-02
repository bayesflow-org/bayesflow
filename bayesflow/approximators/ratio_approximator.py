from collections.abc import Sequence

import keras

from bayesflow.types import Tensor
from bayesflow.adapters import Adapter
from bayesflow.utils import expand_tile, concatenate_valid_shapes
from bayesflow.utils.serialization import serialize, deserialize, serializable

from .approximator import Approximator


@serializable("bayesflow.approximators")
class RatioApproximator(Approximator):
    """Implements NRE-C as described in https://arxiv.org/pdf/2210.06170."""

    def __init__(
        self,
        classifier_network: keras.Layer,
        summary_network: keras.Layer = None,
        gamma: float = 1.0,
        K: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)

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

    def compute_metrics(self, inference_variables: Tensor, inference_conditions: Tensor, stage: str = "training"):
        """TODO"""

        batch_size = keras.ops.shape(inference_variables)[0]

        log_gamma = keras.ops.broadcast_to(keras.ops.log(self.gamma), (batch_size, self.K))
        log_K = keras.ops.broadcast_to(keras.ops.log(self.K), (batch_size, self.K))

        marginal_weight = 1 / (1 + self.gamma)
        joint_weight = self.gamma / (1 + self.gamma)

        # Get (batch_size, K+1, dim) inference variables (theta)
        bootstrap_inference_variables = self.sample_from_batch(inference_variables)
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
        log_numerator_joint = log_gamma + joint_logits
        log_denominator_joint = keras.ops.stack([log_gamma + joint_logits, log_K], axis=-1)
        log_denominator_joint = keras.ops.logsumexp(log_denominator_joint, axis=-1)

        log_numerator_marginal = log_K
        log_denominator_marginal = keras.ops.stack([log_gamma + marginal_logits, log_K], axis=-1)
        log_denominator_marginal = keras.ops.logsumexp(log_denominator_marginal, axis=-1)

        joint_loss = log_denominator_joint - log_numerator_joint
        marginal_loss = log_denominator_marginal - log_numerator_marginal

        loss = marginal_weight * marginal_loss + joint_weight * joint_loss
        loss = keras.ops.mean(loss)

        return {"loss": loss}

    def log_ratio(self, inputs: dict[str, Tensor]):
        inference_variables = inputs.get("inference_variables")
        inference_conditions = inputs.get("inference_conditions")

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
            "summary_network": self.summary_network,
            "classifier_network": self.classifier_network,
            "gamma": self.gamma,
            "K": self.K,
        }

        return base_config | serialize(config)

    def sample_from_batch(self, inference_variables: Tensor, seed=None):
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

        self.built = True

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
