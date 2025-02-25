import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.utils import find_permutation, keras_kwargs, serialize_value_or_type, deserialize_value_or_type

from .actnorm import ActNorm
from .couplings import DualCoupling
from ..inference_network import InferenceNetwork


@serializable(package="networks.coupling_flow")
class CouplingFlow(InferenceNetwork):
    """Implements a coupling flow as a sequence of dual couplings with permutations and activation
    normalization. Incorporates ideas from [1-5].

    [1] Kingma, D. P., & Dhariwal, P. (2018).
    Glow: Generative flow with invertible 1x1 convolutions.
    Advances in Neural Information Processing Systems, 31.

    [2] Durkan, C., Bekasov, A., Murray, I., & Papamakarios, G. (2019).
    Neural spline flows. Advances in Neural Information Processing Systems, 32.

    [3] Ardizzone, L., Kruse, J., Lüth, C., Bracher, N., Rother, C., & Köthe, U. (2020).
    Conditional invertible neural networks for diverse image-to-image translation.
    In DAGM German Conference on Pattern Recognition (pp. 373-387). Springer, Cham.

    [4] Radev, S. T., Mertens, U. K., Voss, A., Ardizzone, L., & Köthe, U. (2020).
    BayesFlow: Learning complex stochastic simulators with invertible neural networks.
    IEEE Transactions on Neural Networks and Learning Systems.

    [5] Alexanderson, S., & Henter, G. E. (2020).
    Robust model training and generalisation with Studentising flows.
    arXiv preprint arXiv:2006.06599.
    """

    def __init__(
        self,
        depth: int = 6,
        subnet: str | type = "mlp",
        transform: str = "affine",
        permutation: str | None = "random",
        use_actnorm: bool = True,
        base_distribution: str = "normal",
        **kwargs,
    ):
        super().__init__(base_distribution=base_distribution, **keras_kwargs(kwargs))

        self.depth = depth

        self.invertible_layers = []
        for i in range(depth):
            if use_actnorm:
                self.invertible_layers.append(ActNorm(**kwargs.get("actnorm_kwargs", {})))

            if (p := find_permutation(permutation, **kwargs.get("permutation_kwargs", {}))) is not None:
                self.invertible_layers.append(p)

            self.invertible_layers.append(DualCoupling(subnet, transform, **kwargs.get("coupling_kwargs", {})))

        # serialization: store all parameters necessary to call __init__
        self.config = {
            "depth": depth,
            "transform": transform,
            "permutation": permutation,
            "use_actnorm": use_actnorm,
            "base_distribution": base_distribution,
            **kwargs,
        }
        self.config = serialize_value_or_type(self.config, "subnet", subnet)

    # noinspection PyMethodOverriding
    def build(self, xz_shape, conditions_shape=None):
        super().build(xz_shape)

        for layer in self.invertible_layers:
            layer.build(xz_shape=xz_shape, conditions_shape=conditions_shape)

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config

    @classmethod
    def from_config(cls, config):
        config = deserialize_value_or_type(config, "subnet")
        return cls(**config)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        z = x
        log_det = keras.ops.zeros(keras.ops.shape(x)[:-1])
        for layer in self.invertible_layers:
            z, det = layer(z, conditions=conditions, inverse=False, training=training, **kwargs)
            log_det += det

        if density:
            log_density_latent = self.base_distribution.log_prob(z)
            log_density = log_density_latent + log_det
            return z, log_density

        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        x = z
        log_det = keras.ops.zeros(keras.ops.shape(z)[:-1])
        for layer in reversed(self.invertible_layers):
            x, det = layer(x, conditions=conditions, inverse=True, training=training, **kwargs)
            log_det += det

        if density:
            log_prob = self.base_distribution.log_prob(z)
            log_density = log_prob - log_det
            return x, log_density

        return x

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, sample_weights: Tensor = None, stage: str = "training"
    ) -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(x, conditions=conditions, sample_weights=sample_weights, stage=stage)

        z, log_density = self(x, conditions=conditions, inverse=False, density=True)
        loss = self.aggregate(-log_density, sample_weights)

        return base_metrics | {"loss": loss}
