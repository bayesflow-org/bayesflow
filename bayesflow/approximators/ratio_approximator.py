from collections.abc import Sequence

import keras

from bayesflow.types import Tensor
from bayesflow.adapters import Adapter
from bayesflow.utils.serialization import serialize, deserialize, serializable

from .approximator import Approximator


@serializable("bayesflow.approximators")
class RatioApproximator(Approximator):
    """
    Implements NRE-C as described in https://arxiv.org/pdf/2210.06170.
    """

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
        self.K = float(K)

        self.projector = keras.layers.Dense(units=1)

        self.seed_generator = keras.random.SeedGenerator()

    def compute_metrics(self, inference_variables: Tensor, inference_conditions: Tensor, stage: str = "training"):
        """
        :param inference_variables: Tensor of shape (batch_size, param_dim)
            True inference_variables Θ, sampled from the joint p(x, Θ)
        :param inference_conditions: Tensor of shape (batch_size, n_obs, obs_dim)
            True inference_conditions x, sampled from the joint p(x, Θ)
        :return:
        """

        batch_size = keras.ops.shape(inference_variables)[0]

        log_gamma = keras.ops.broadcast_to(keras.ops.log(self.gamma), batch_size)
        log_K = keras.ops.broadcast_to(keras.ops.log(self.K), batch_size)

        joint_log_ratio = self.log_ratio(inference_variables, inference_conditions, stage=stage)
        marginal_log_ratio = self.log_ratio(
            keras.random.shuffle(inference_variables, axis=0, seed=self.seed_generator), inference_conditions
        )

        # Eq. 7 - we use a trick for numerical stability:
        # log(K + gamma * sum_{i=1}^{K} exp(h_i)) = log(exp(log K) + sum_{i=1}^{K} exp(h_i + log gamma))
        # so if we absorb log gamma into the network outputs and concatenate log K, we can use logsumexp
        log_numerator_joint = log_gamma + joint_log_ratio
        log_denominator_joint = keras.ops.stack([log_gamma + joint_log_ratio, log_K], axis=-1)
        log_denominator_joint = keras.ops.logsumexp(log_denominator_joint, axis=-1)

        log_numerator_marginal = log_K
        log_denominator_marginal = keras.ops.stack([log_gamma + marginal_log_ratio, log_K], axis=-1)
        log_denominator_marginal = keras.ops.logsumexp(log_denominator_marginal, axis=-1)

        joint_loss = log_denominator_joint - log_numerator_joint
        marginal_loss = log_denominator_marginal - log_numerator_marginal

        marginal_weight = 1 / (1 + self.gamma)
        joint_weight = self.gamma / (1 + self.gamma)

        # Eq. 9
        loss = marginal_weight * marginal_loss + joint_weight * joint_loss
        loss = keras.ops.mean(loss, axis=0)

        return {"loss": loss}

    def log_ratio(self, inference_variables: Tensor, inference_conditions: Tensor, stage: str):
        if self.summary_network is not None:
            inference_conditions = self.summary_network(inference_conditions, training=stage == "training")

        classifier_inputs = keras.ops.concatenate([inference_variables, inference_conditions], axis=-1)

        log_ratio = self.classifier_network(classifier_inputs, training=stage == "training")
        log_ratio = self.projector(log_ratio)
        log_ratio = keras.ops.squeeze(log_ratio, axis=-1)
        return log_ratio

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

    def build(self, data_shapes):
        if self.built:
            return

        self.log_ratio(
            keras.ops.zeros(data_shapes["inference_variables"]), keras.ops.zeros(data_shapes["inference_conditions"])
        )
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
