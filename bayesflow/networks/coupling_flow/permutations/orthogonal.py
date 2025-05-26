from keras import ops

from bayesflow.types import Shape, Tensor
from bayesflow.utils.serialization import serializable

from ..invertible_layer import InvertibleLayer


@serializable("bayesflow.networks")
class OrthogonalPermutation(InvertibleLayer):
    """Implements a learnable orthogonal transformation according to [1]. Can be
    used as an alternative to a fixed ``Permutation`` layer.

    [1] Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1x1
    convolutions. Advances in neural information processing systems, 31.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight = None

    def build(self, xz_shape: Shape, **kwargs) -> None:
        self.weight = self.add_weight(shape=(xz_shape[-1], xz_shape[-1]), initializer="orthogonal", trainable=True)

    def call(self, xz: Tensor, inverse: bool = False, **kwargs) -> (Tensor, Tensor):
        if inverse:
            return self._inverse(xz)
        return self._forward(xz)

    def _forward(self, x: Tensor) -> (Tensor, Tensor):
        z = ops.matmul(x, self.weight)
        log_det = ops.log(ops.abs(ops.det(self.weight)))

        if ops.ndim(x) > 2:
            log_det = ops.multiply(log_det, ops.shape(x)[1])

        return z, log_det

    def _inverse(self, z: Tensor) -> (Tensor, Tensor):
        weight = ops.inv(self.weight)

        x = ops.matmul(z, weight)
        log_det = ops.log(ops.abs(ops.det(weight)))

        if ops.ndim(z) > 2:
            log_det = ops.multiply(log_det, ops.shape(z)[1])

        return x, log_det
