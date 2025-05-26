import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils.serialization import serializable

from bayesflow.networks.inference_network import InferenceNetwork
from bayesflow.networks.coupling_flow import CouplingFlow

from .conditional_gaussian import ConditionalGaussian


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class CIF(InferenceNetwork):
    """(IN) Implements a continuously indexed flow (CIF) with a `CouplingFlow`
    bijection and `ConditionalGaussian` distributions p and q. Improves on
    eliminating leaky sampling found topologically in normalizing flows.
    Built in reference to [1].

    [1] R. Cornish, A. Caterini, G. Deligiannidis, & A. Doucet (2021).
    Relaxing Bijectivity Constraints with Continuously Indexed Normalising
    Flows.
    arXiv:1909.13833.
    """

    def __init__(self, pq_depth: int = 4, pq_width: int = 128, pq_activation: str = "swish", **kwargs):
        """Creates an instance of a `CIF` with configurable
        `ConditionalGaussian` distributions p and q, each containing MLP
        networks

        Parameters
        ----------
        pq_depth: int, optional, default: 4
            The number of MLP hidden layers (minimum: 1)
        pq_width: int, optional, default: 128
            The dimensionality of the MLP hidden layers
        pq_activation: str, optional, default: 'tanh'
            The MLP activation function
        """

        super().__init__(base_distribution="normal", **kwargs)
        self.bijection = CouplingFlow()
        self.p_dist = ConditionalGaussian(depth=pq_depth, width=pq_width, activation=pq_activation)
        self.q_dist = ConditionalGaussian(depth=pq_depth, width=pq_width, activation=pq_activation)

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        super().build(xz_shape)
        self.bijection.build(xz_shape, conditions_shape=conditions_shape)
        self.p_dist.build(xz_shape)
        self.q_dist.build(xz_shape)

    def call(
        self, xz: Tensor, conditions: Tensor = None, inverse: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        if inverse:
            return self._inverse(xz, conditions=conditions, **kwargs)
        return self._forward(xz, conditions=conditions, **kwargs)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        # Sample u ~ q_u
        u, log_qu = self.q_dist.sample(x, log_prob=True)

        # Bijection and log Jacobian x -> z
        z, log_jac = self.bijection(x, conditions=conditions, density=True)
        if log_jac.ndim > 1:
            log_jac = keras.ops.sum(log_jac, axis=1)

        # Log prob over p on u with conditions z
        log_pu = self.p_dist.log_prob(u, z)

        # Prior log prob
        log_prior = self.base_distribution.log_prob(z)
        if log_prior.ndim > 1:
            log_prior = keras.ops.sum(log_prior, axis=1)

        # we cannot compute an exact analytical density
        elbo = log_jac + log_pu + log_prior - log_qu

        if density:
            return z, elbo

        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        if not density:
            return self.bijection(z, conditions=conditions, inverse=True, density=False)

        u = self.p_dist.sample(z)
        x = self.bijection(z, conditions=conditions, inverse=True)

        log_pu = self.p_dist.log_prob(u, x)

        return x, log_pu

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training") -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(x, conditions=conditions, stage=stage)

        elbo = self.log_prob(x, conditions=conditions)

        loss = -keras.ops.mean(elbo)

        return base_metrics | {"loss": loss}
