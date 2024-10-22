import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.utils import find_permutation, keras_kwargs

from .actnorm import ActNorm
from .couplings import DualCoupling
from ..inference_network import InferenceNetwork


@serializable(package="bayesflow.networks")
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
            if (p := find_permutation(permutation, **kwargs.get("permutation_kwargs", {}))) is not None:
                self.invertible_layers.append(p)

            self.invertible_layers.append(DualCoupling(subnet, transform, **kwargs.get("coupling_kwargs", {})))

            if use_actnorm:
                self.invertible_layers.append(ActNorm(**kwargs.get("actnorm_kwargs", {})))

    # noinspection PyMethodOverriding
    def build(self, xz_shape, conditions_shape=None):
        super().build(xz_shape)

        for layer in self.invertible_layers:
            layer.build(xz_shape=xz_shape, conditions_shape=conditions_shape)

    def call(
        self, xz: Tensor, conditions: Tensor = None, inverse: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        if inverse:
            return self._inverse(xz, conditions=conditions, **kwargs)
        return self._forward(xz, conditions=conditions, **kwargs)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        z = x
        log_det = keras.ops.zeros(keras.ops.shape(x)[:-1])
        for layer in self.invertible_layers:
            z, det = layer(z, conditions=conditions, inverse=False, **kwargs)
            log_det += det

        if density:
            log_prob = self.base_distribution.log_prob(z)
            log_density = log_prob + log_det
            return z, log_density

        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        x = z
        log_det = keras.ops.zeros(keras.ops.shape(z)[:-1])
        for layer in reversed(self.invertible_layers):
            x, det = layer(x, conditions=conditions, inverse=True, **kwargs)
            log_det += det

        if density:
            log_prob = self.base_distribution.log_prob(z)
            log_density = log_prob - log_det
            return x, log_density

        return x

    def compute_metrics(self, x: Tensor, conditions: Tensor = None, stage: str = "training") -> dict[str, Tensor]:
        base_metrics = super().compute_metrics(x, conditions=conditions, stage=stage)

        z, log_density = self(x, conditions=conditions, inverse=False, density=True)
        loss = -keras.ops.mean(log_density)

        return base_metrics | {"loss": loss}
