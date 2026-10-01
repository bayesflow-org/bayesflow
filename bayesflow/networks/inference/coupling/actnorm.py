from keras import ops

from bayesflow.types import Shape, Tensor
from bayesflow.utils.serialization import serializable

from .invertible_layer import InvertibleLayer


@serializable("bayesflow.networks")
class ActNorm(InvertibleLayer):
    """Implements an Activation Normalization (ActNorm) Layer. Activation Normalization is learned invertible
    normalization, using a scale (s) and a bias (b) vector::

       y = s * x + b(forward)
       x = (y - b) / s(inverse)

    References
    ----------

    [1] Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with invertible 1x1 convolutions.
        Advances in Neural Information Processing Systems, 31.

    [2] Salimans, Tim, and Durk P. Kingma. (2016). Weight normalization: A simple reparameterization to accelerate
       training of deep neural networks. Advances in Neural Information Processing Systems, 29.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.scale = None
        self.bias = None

    def build(self, xz_shape: Shape, **kwargs):
        self.scale = self.add_weight(shape=(xz_shape[-1],), initializer="ones", name="scale")
        self.bias = self.add_weight(shape=(xz_shape[-1],), initializer="zeros", name="bias")

    def call(self, xz: Tensor, inverse: bool = False, **kwargs) -> tuple[Tensor, Tensor]:
        if inverse:
            return self._inverse(xz, **kwargs)
        return self._forward(xz, **kwargs)

    def _forward(self, x: Tensor, fixed_target_mask: Tensor = None, **kwargs) -> tuple[Tensor, Tensor]:
        z = self.scale * x + self.bias
        log_jac = ops.log(ops.abs(self.scale))
        z, log_jac = self._skip_fixed_dims(x, z, log_jac, fixed_target_mask)
        log_det = ops.sum(log_jac, axis=-1)
        log_det = ops.broadcast_to(log_det, ops.shape(x)[:-1])
        return z, log_det

    def _inverse(self, z: Tensor, fixed_target_mask: Tensor = None, **kwargs) -> tuple[Tensor, Tensor]:
        x = (z - self.bias) / self.scale
        log_jac = -ops.log(ops.abs(self.scale))
        x, log_jac = self._skip_fixed_dims(z, x, log_jac, fixed_target_mask)
        log_det = ops.sum(log_jac, axis=-1)
        log_det = ops.broadcast_to(log_det, ops.shape(z)[:-1])
        return x, log_det
