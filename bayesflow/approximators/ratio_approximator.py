from collections.abc import Sequence
import keras

from bayesflow.adapters import Adapter
from bayesflow.utils.serialization import serialize, deserialize, serializable
from .approximator import Approximator


@serializable("bayesflow.approximators")
class RatioApproximator(Approximator):
    """
    Implements all three Neural Ratio Estimation procedures as detailed in https://arxiv.org/pdf/2210.06170
    NRE-A and NRE-B are implemented as a special case of NRE-C, by setting:
    NRE-A: gamma = 1, K = 1
    NRE-B: gamma = infinity
    """

    def __init__(self, summary_network, classifier_network, gamma, K, **kwargs):
        super().__init__(**kwargs)
        self.summary_network = summary_network
        self.classifier_network = classifier_network
        self.projector = keras.layers.Dense(units=1)

        if gamma <= 0:
            raise ValueError(f"Gamma must be positive, got {gamma}.")
        if gamma == float("inf"):
            raise NotImplementedError("NRE-B is not yet supported.")

        if K <= 0:
            raise ValueError(f"K must be positive, got {K}.")

        self.gamma = gamma
        self.K = K

        self.seed_generator = keras.random.SeedGenerator()

    @classmethod
    def nre_a(cls, summary_network, classifier_network, **kwargs):
        """Initialize the approximator to run NRE-A."""
        return cls(summary_network, classifier_network, gamma=1, K=1, **kwargs)

    @classmethod
    def nre_b(cls, summary_network, classifier_network, K=5, **kwargs):
        """Initialize the approximator to run NRE-B."""
        return cls(summary_network, classifier_network, gamma=float("inf"), K=K, **kwargs)

    @classmethod
    def nre_c(cls, summary_network, classifier_network, gamma=5, K=5, **kwargs):
        """Initialize the approximator to run NRE-C."""
        return cls(summary_network, classifier_network, gamma=gamma, K=K, **kwargs)

    def compute_metrics(self, parameters, observables):
        """
        :param parameters: Tensor of shape (batch_size, param_dim)
            True parameters Θ, sampled from the joint p(x, Θ)
        :param observables: Tensor of shape (batch_size, n_obs, obs_dim)
            True observables x, sampled from the joint p(x, Θ)
        :return:
        """
        batch_size = keras.ops.shape(parameters)[0]

        joint_parameters = parameters
        joint_observables = observables
        joint_log_ratio = self.log_ratio(joint_parameters, joint_observables)

        marginal_parameters = keras.random.shuffle(parameters, axis=0, seed=self.seed_generator)
        marginal_observables = observables
        marginal_log_ratio = self.log_ratio(marginal_parameters, marginal_observables)

        # FIXME: NRE-B may need its own implementation
        log_gamma = keras.ops.broadcast_to(keras.ops.log(self.gamma), (batch_size,))
        log_K = keras.ops.broadcast_to(keras.ops.log(self.K), (batch_size,))

        # Eq. 7
        log_numerator_joint = log_gamma + joint_log_ratio
        log_numerator_marginal = log_K

        # Eq. 7 - we use a trick for numerical stability:
        # log(K + gamma * sum_{i=1}^{K} exp(h_i)) = log(exp(log K) + sum_{i=1}^{K} exp(h_i + log gamma))
        # so if we absorb log gamma into the network outputs and concatenate log K, we can use logsumexp
        log_denominator_joint = keras.ops.stack([log_gamma + joint_log_ratio, log_K], axis=1)
        log_denominator_joint = keras.ops.logsumexp(log_denominator_joint, axis=1)

        log_denominator_marginal = keras.ops.stack([log_gamma + marginal_log_ratio, log_K], axis=1)
        log_denominator_marginal = keras.ops.logsumexp(log_denominator_marginal, axis=1)

        joint_loss = log_denominator_joint - log_numerator_joint
        marginal_loss = log_denominator_marginal - log_numerator_marginal

        if self.gamma == float("inf"):
            marginal_weight = 0
            joint_weight = 1
        else:
            marginal_weight = 1 / (1 + self.gamma)
            joint_weight = self.gamma / (1 + self.gamma)

        # Eq. 9
        loss = marginal_weight * marginal_loss + joint_weight * joint_loss
        loss = keras.ops.mean(loss, axis=0)

        return {"loss": loss}

    def log_ratio(self, parameters, observables):
        """
        :param parameters: Tensor of shape (batch_size, param_dim)
            Parameters Θ
        :param observables: Tensor of shape (batch_size, n_obs, obs_dim)
            Observables x
        :return: Tensor of shape (batch_size,)
            NRE-A: The ratio log p(x | Θ) - log p(x)
            NRE-B: The ratio log p(Θ | x) - log p(Θ) + c(x)
            NRE-C: The ratio log p(x | Θ) - log p(x)
        """
        observables = self.summary_network(observables)
        log_ratio = self.classifier_network(keras.ops.concatenate([parameters, observables], axis=1))
        log_ratio = self.projector(log_ratio)
        log_ratio = keras.ops.squeeze(log_ratio, axis=1)
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
        self.log_ratio(keras.ops.zeros(data_shapes["parameters"]), keras.ops.zeros(data_shapes["observables"]))
        self.built = True

    @classmethod
    def build_adapter(cls, parameter_names: str | Sequence[str], observables_names: str | Sequence[str]):
        if isinstance(parameter_names, str):
            parameter_names = [parameter_names]
        if isinstance(observables_names, str):
            observables_names = [observables_names]

        adapter = Adapter()
        adapter.to_array()
        adapter.convert_dtype("float64", "float32")
        adapter.concatenate(parameter_names, into="parameters")
        adapter.concatenate(observables_names, into="observables")
        return adapter

    def _batch_size_from_data(self, data: any) -> int:
        return keras.ops.shape(data["parameters"])[0]
