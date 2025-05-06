import keras
import numpy as np
from bayesflow.networks.mlp import MLP

from bayesflow.types import Shape, Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable


# disable module check, use potential module after moving from experimental
@serializable("bayesflow.networks", disable_module_check=True)
class ConditionalGaussian(keras.Layer):
    """Implements a conditional gaussian distribution with neural networks for
    the means and standard deviations respectively. Bulit in reference to [1].

    [1] R. Cornish, A. Caterini, G. Deligiannidis, & A. Doucet (2021).
    Relaxing Bijectivity Constraints with Continuously Indexed Normalising
    Flows.
    arXiv:1909.13833.
    """

    def __init__(self, depth: int = 4, width: int = 128, activation: str = "swish", **kwargs):
        """Creates an instance of a `ConditionalGaussian` with configurable
        `MLP` networks for the means and standard deviations.

        Parameters
        ----------
        depth: int, optional, default: 4
            The number of MLP hidden layers (minimum: 1)
        width: int, optional, default: 128
            The dimensionality of the MLP hidden layers
        activation: str, optional, default: "swish"
            The MLP activation function
        """

        super().__init__(**layer_kwargs(kwargs))
        self.means = MLP([width] * depth, activation=activation)
        self.stds = MLP([width] * depth, activation=activation)
        self.output_projector = keras.layers.Dense(None)

    def build(self, input_shape: Shape) -> None:
        self.means.build(input_shape)
        self.stds.build(input_shape)
        self.output_projector.units = input_shape[-1]

    def _diagonal_gaussian_log_prob(self, conditions: Tensor, means: Tensor, stds: Tensor) -> Tensor:
        batch_size = keras.ops.shape(conditions)[0]

        if keras.ops.shape(means)[0] != batch_size or keras.ops.shape(stds)[0] != batch_size:
            raise ValueError("Means and stds must have the same batch size as conditions.")

        flat_conditions = keras.ops.reshape(conditions, (batch_size, -1))
        flat_means = keras.ops.reshape(means, (batch_size, -1))
        flat_stds = keras.ops.reshape(stds, (batch_size, -1))

        flat_variances = flat_stds**2

        dim = keras.ops.shape(flat_conditions)[1]

        const_term = -0.5 * dim * np.log(2 * np.pi)
        log_det_terms = -0.5 * keras.ops.sum(keras.ops.log(flat_variances), axis=1)
        product_terms = -0.5 * keras.ops.sum((flat_conditions - flat_means) ** 2 / flat_variances, axis=1)

        return const_term + log_det_terms + product_terms

    def log_prob(self, x: Tensor, conditions: Tensor) -> Tensor:
        means = self.output_projector(self.means(conditions))
        stds = keras.ops.exp(self.output_projector(self.stds(conditions)))
        return self._diagonal_gaussian_log_prob(x, means, stds)

    def sample(self, conditions: Tensor, log_prob: bool = False) -> Tensor | tuple[Tensor, Tensor]:
        means = self.output_projector(self.means(conditions))
        stds = keras.ops.exp(self.output_projector(self.stds(conditions)))

        # re-parametrize
        samples = stds * keras.random.normal(keras.ops.shape(conditions)) + means

        if log_prob:
            log_p = self._diagonal_gaussian_log_prob(samples, means, stds)
            return samples, log_p

        return samples
