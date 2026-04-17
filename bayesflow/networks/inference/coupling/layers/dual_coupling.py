import keras

from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, serialize
from bayesflow.types import Shape, Tensor

from .single_coupling import SingleCoupling

from ..invertible_layer import InvertibleLayer


@serializable("bayesflow.networks")
class DualCoupling(InvertibleLayer):
    """Dual coupling layer composed of two sequential single coupling layers.

    Implements a coupling transformation by alternately transforming the two halves
    of the input, realizing a normalizing flow.

    The layer splits input into two parts and applies two sequential single coupling
    transformations, doubling back on the transformed variables for each coupling.

    Parameters
    ----------
    subnet : str or type, optional
        A neural network type for the coupling subnet. If a string, should be a
        registered name (e.g., "mlp"). If a type, will be instantiated with the
        provided kwargs. Default is "mlp".
    transform : str, optional
        Name of the transformation to apply (e.g., "affine"). Default is "affine".
    **kwargs
        Additional keyword arguments passed to `InvertibleLayer` and propagated
        to individual `SingleCoupling` layers.
    """

    def __init__(self, subnet: str | type = "mlp", transform: str = "affine", **kwargs):
        super().__init__(**kwargs)
        self.subnet = subnet
        self.transform = transform

        self.coupling1 = SingleCoupling(subnet, transform, **kwargs)
        self.coupling2 = SingleCoupling(subnet, transform, **kwargs)
        self.pivot = None

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "subnet": self.subnet,
            "transform": self.transform,
        }

        return base_config | serialize(config)

    def build(self, xz_shape: Shape, conditions_shape: Shape = None):
        xz_shape = tuple(xz_shape)
        if conditions_shape is not None:
            conditions_shape = tuple(conditions_shape)
        self.pivot = xz_shape[-1] // 2

        x1_shape = xz_shape[:1] + (self.pivot,)
        x2_shape = xz_shape[:1] + (xz_shape[-1] - self.pivot,)

        self.coupling1.build(x1_shape, x2_shape, conditions_shape)
        self.coupling2.build(x2_shape, x1_shape, conditions_shape)

    def call(
        self, xz: Tensor, conditions: Tensor = None, inverse: bool = False, training: bool = False, **kwargs
    ) -> tuple[Tensor, Tensor]:
        if inverse:
            return self._inverse(xz, conditions=conditions, training=training, **kwargs)
        return self._forward(xz, conditions=conditions, training=training, **kwargs)

    def _forward(self, x: Tensor, conditions: Tensor = None, training: bool = False, **kwargs) -> tuple[Tensor, Tensor]:
        """Transform (x1, x2) -> (g(x1; f(x2; x1)), f(x2; x1))"""
        x1, x2 = x[..., : self.pivot], x[..., self.pivot :]
        (z1, z2), log_det1 = self.coupling1(x1, x2, conditions=conditions, training=training, **kwargs)
        (z2, z1), log_det2 = self.coupling2(z2, z1, conditions=conditions, training=training, **kwargs)

        z = keras.ops.concatenate([z1, z2], axis=-1)
        log_det = log_det1 + log_det2

        return z, log_det

    def _inverse(self, z: Tensor, conditions: Tensor = None, training: bool = False, **kwargs) -> tuple[Tensor, Tensor]:
        """Transform (g(x1; f(x2; x1)), f(x2; x1)) -> (x1, x2)"""
        z1, z2 = z[..., : self.pivot], z[..., self.pivot :]
        (z2, z1), log_det2 = self.coupling2(z2, z1, conditions=conditions, inverse=True, training=training, **kwargs)
        (x1, x2), log_det1 = self.coupling1(z1, z2, conditions=conditions, inverse=True, training=training, **kwargs)

        x = keras.ops.concatenate([x1, x2], axis=-1)
        log_det = log_det1 + log_det2

        return x, log_det
