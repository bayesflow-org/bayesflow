import keras

from keras.saving import register_keras_serializable as serializable

from bayesflow.types import Tensor
from bayesflow.utils import find_network, keras_kwargs
from ..invertible_layer import InvertibleLayer
from ..transforms import find_transform


@serializable(package="bayesflow.networks.coupling_flow")
class SingleCoupling(InvertibleLayer):
    """
    Implements a single coupling layer as a composition of a subnet and a transform.

    Subnet output tensors are linearly mapped to the correct dimension.
    """

    def __init__(self, subnet: str | type = "mlp", transform: str = "affine", **kwargs):
        super().__init__(**keras_kwargs(kwargs))

        self.network = find_network(subnet, **kwargs.get("subnet_kwargs", {}))
        self.transform = find_transform(transform, **kwargs.get("transform_kwargs", {}))

        output_projector_kwargs = kwargs.get("output_projector_kwargs", {})
        output_projector_kwargs.setdefault("kernel_initializer", "zeros")
        self.output_projector = keras.layers.Dense(units=None, **output_projector_kwargs)

    # noinspection PyMethodOverriding
    def build(self, x1_shape, x2_shape, conditions_shape=None):
        self.output_projector.units = self.transform.params_per_dim * x2_shape[-1]

        x1 = keras.ops.zeros(x1_shape)
        x2 = keras.ops.zeros(x2_shape)
        if conditions_shape is None:
            conditions = None
        else:
            conditions = keras.ops.zeros(conditions_shape)

        # build nested layers with forward pass
        self.call(x1, x2, conditions=conditions)

    def call(
        self, x1: Tensor, x2: Tensor, conditions: Tensor = None, inverse: bool = False, **kwargs
    ) -> ((Tensor, Tensor), Tensor):
        if inverse:
            return self._inverse(x1, x2, conditions=conditions, **kwargs)
        return self._forward(x1, x2, conditions=conditions, **kwargs)

    def _forward(self, x1: Tensor, x2: Tensor, conditions: Tensor = None, **kwargs) -> ((Tensor, Tensor), Tensor):
        """Transform (x1, x2) -> (x1, f(x2; x1))"""
        z1 = x1
        parameters = self.get_parameters(x1, conditions=conditions, **kwargs)
        z2, log_det = self.transform(x2, parameters=parameters)

        return (z1, z2), log_det

    def _inverse(self, z1: Tensor, z2: Tensor, conditions: Tensor = None, **kwargs) -> ((Tensor, Tensor), Tensor):
        """Transform (x1, f(x2; x1)) -> (x1, x2)"""
        x1 = z1
        parameters = self.get_parameters(x1, conditions=conditions, **kwargs)
        x2, log_det = self.transform(z2, parameters=parameters, inverse=True)

        return (x1, x2), log_det

    def get_parameters(self, x: Tensor, conditions: Tensor = None, **kwargs) -> dict[str, Tensor]:
        if conditions is not None:
            x = keras.ops.concatenate([x, conditions], axis=-1)

        parameters = self.output_projector(self.network(x, **kwargs))
        parameters = self.transform.split_parameters(parameters)
        parameters = self.transform.constrain_parameters(parameters)

        return parameters
