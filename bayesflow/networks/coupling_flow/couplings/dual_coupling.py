import keras
from keras.saving import register_keras_serializable as serializable

from bayesflow.utils import keras_kwargs, serialize_value_or_type, deserialize_value_or_type
from bayesflow.types import Tensor
from .single_coupling import SingleCoupling
from ..invertible_layer import InvertibleLayer


@serializable(package="networks.coupling_flow")
class DualCoupling(InvertibleLayer):
    def __init__(self, subnet: str | type = "mlp", transform: str = "affine", **kwargs):
        super().__init__(**keras_kwargs(kwargs))
        self.coupling1 = SingleCoupling(subnet, transform, **kwargs)
        self.coupling2 = SingleCoupling(subnet, transform, **kwargs)
        self.pivot = None

        # serialization: store all parameters necessary to call __init__
        self.config = {
            "transform": transform,
            **kwargs,
        }
        self.config = serialize_value_or_type(self.config, "subnet", subnet)

    def get_config(self):
        base_config = super().get_config()
        return base_config | self.config

    @classmethod
    def from_config(cls, config):
        config = deserialize_value_or_type(config, "subnet")
        return cls(**config)

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
