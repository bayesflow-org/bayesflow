import keras

from bayesflow.utils import model_kwargs
from bayesflow.utils.serialization import deserialize, serializable, serialize
from bayesflow.types import Tensor

from .single_coupling import SingleCoupling

from ..invertible_layer import InvertibleLayer


@serializable("bayesflow.networks")
class DualCoupling(InvertibleLayer):
    def __init__(self, subnet: str | type = "mlp", transform: str = "affine", **kwargs):
        super().__init__(**kwargs)
        self.subnet = subnet
        self.transform = transform

        self.coupling1 = SingleCoupling(subnet, transform, **kwargs)
        self.coupling2 = SingleCoupling(subnet, transform, **kwargs)
        self.pivot = None

    def get_config(self):
        base_config = super().get_config()
        base_config = model_kwargs(base_config)

        config = {
            "subnet": self.subnet,
            "transform": self.transform,
        }

        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    # noinspection PyMethodOverriding
    def build(self, xz_shape, conditions_shape=None):
        self.pivot = xz_shape[-1] // 2

        xz = keras.ops.zeros(xz_shape)
        if conditions_shape is None:
            conditions = None
        else:
            conditions = keras.ops.zeros(conditions_shape)

        # build nested layers with forward pass
        self.call(xz, conditions=conditions)

    def call(
        self, xz: Tensor, conditions: Tensor = None, inverse: bool = False, training: bool = False, **kwargs
    ) -> (Tensor, Tensor):
        if inverse:
            return self._inverse(xz, conditions=conditions, training=training, **kwargs)
        return self._forward(xz, conditions=conditions, training=training, **kwargs)

    def _forward(self, x: Tensor, conditions: Tensor = None, training: bool = False, **kwargs) -> (Tensor, Tensor):
        """Transform (x1, x2) -> (g(x1; f(x2; x1)), f(x2; x1))"""
        x1, x2 = x[..., : self.pivot], x[..., self.pivot :]
        (z1, z2), log_det1 = self.coupling1(x1, x2, conditions=conditions, training=training, **kwargs)
        (z2, z1), log_det2 = self.coupling2(z2, z1, conditions=conditions, training=training, **kwargs)

        z = keras.ops.concatenate([z1, z2], axis=-1)
        log_det = log_det1 + log_det2

        return z, log_det

    def _inverse(self, z: Tensor, conditions: Tensor = None, training: bool = False, **kwargs) -> (Tensor, Tensor):
        """Transform (g(x1; f(x2; x1)), f(x2; x1)) -> (x1, x2)"""
        z1, z2 = z[..., : self.pivot], z[..., self.pivot :]
        (z2, z1), log_det2 = self.coupling2(z2, z1, conditions=conditions, inverse=True, training=training, **kwargs)
        (x1, x2), log_det1 = self.coupling1(z1, z2, conditions=conditions, inverse=True, training=training, **kwargs)

        x = keras.ops.concatenate([x1, x2], axis=-1)
        log_det = log_det1 + log_det2

        return x, log_det
