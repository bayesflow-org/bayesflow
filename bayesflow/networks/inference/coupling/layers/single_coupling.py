from typing import Any

import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import filter_kwargs, find_network, layer_kwargs, concatenate_valid
from bayesflow.utils.serialization import serializable, serialize

from ..invertible_layer import InvertibleLayer
from ..transforms import find_transform

from ....defaults import COUPLING_MLP_DEFAULTS


@serializable("bayesflow.networks")
class SingleCoupling(InvertibleLayer):
    """Implements a single coupling layer as a composition of a subnet and a transform.

    A coupling layer partitions input into two parts: one part remains unchanged,
    while the other is transformed via a parametric transformation whose parameters
    are computed by a neural network (subnet) applied to the unchanged part.

    Parameters
    ----------
    subnet : str or type, optional
        A neural network type for computing transformation parameters. If a string,
        should be a registered name (e.g., "mlp"). If a type, will be instantiated
        with the provided *subnet_kwargs*. Default is "mlp".
    transform : str, optional
        Name of the transformation to apply (e.g., "affine"). Default is "affine".
    subnet_kwargs : dict[str, Any], optional
        Keyword arguments passed to the subnet constructor or used to update the
        default subnet settings. Default is None.
    transform_kwargs : dict[str, Any], optional
        Keyword arguments passed to the transform constructor. Default is None.
    **kwargs
        Additional keyword arguments passed to `InvertibleLayer`.
    """

    def __init__(
        self,
        subnet: str | type = "mlp",
        transform: str = "affine",
        subnet_kwargs: dict[str, Any] = None,
        transform_kwargs: dict[str, Any] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        subnet_kwargs = subnet_kwargs or {}
        transform_kwargs = transform_kwargs or {}

        if subnet == "mlp":
            subnet_kwargs = COUPLING_MLP_DEFAULTS | subnet_kwargs

        self.subnet = find_network(subnet, **subnet_kwargs)
        self.transform = find_transform(transform, **transform_kwargs)
        self.output_projector = None

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "subnet": self.subnet,
            "transform": self.transform,
        }

        return base_config | serialize(config)

    # noinspection PyMethodOverriding
    def build(self, x1_shape: Shape, x2_shape: Shape, conditions_shape: Shape = None):
        self.output_projector = keras.layers.Dense(
            units=self.transform.params_per_dim * x2_shape[-1],
            kernel_initializer="zeros",
            bias_initializer="zeros",
            name="output_projector",
        )

        if conditions_shape is not None:
            subnet_input_shape = tuple(x1_shape[:-1]) + (x1_shape[-1] + conditions_shape[-1],)
        else:
            subnet_input_shape = tuple(x1_shape)

        self.subnet.build(subnet_input_shape)
        out_shape = self.subnet.compute_output_shape(subnet_input_shape)
        self.output_projector.build(out_shape)
        self.transform.build(x2_shape)

    def call(
        self, x1: Tensor, x2: Tensor, conditions: Tensor = None, inverse: bool = False, training: bool = False, **kwargs
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        if inverse:
            return self._inverse(x1, x2, conditions=conditions, training=training, **kwargs)
        return self._forward(x1, x2, conditions=conditions, training=training, **kwargs)

    def _forward(
        self, x1: Tensor, x2: Tensor, conditions: Tensor = None, training: bool = False, **kwargs
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        """Transform (x1, x2) -> (x1, f(x2; x1))"""
        z1 = x1
        parameters = self.get_parameters(x1, conditions=conditions, training=training)
        z2, log_det = self.transform(x2, parameters=parameters)

        return (z1, z2), log_det

    def _inverse(
        self, z1: Tensor, z2: Tensor, conditions: Tensor = None, training: bool = False, **kwargs
    ) -> tuple[tuple[Tensor, Tensor], Tensor]:
        """Transform (x1, f(x2; x1)) -> (x1, x2)"""
        x1 = z1
        parameters = self.get_parameters(x1, conditions=conditions, training=training, **kwargs)
        x2, log_det = self.transform(z2, parameters=parameters, inverse=True)

        return (x1, x2), log_det

    def get_parameters(
        self, x: Tensor, conditions: Tensor = None, training: bool = False, **kwargs
    ) -> dict[str, Tensor]:
        """Applies the inner neural network to obtain the transformation parameters, for instance,
        if affine transformations, then [s, t] = NN(inputs), followed by a constraint, e.g., s = exp(s).
        """
        inputs = concatenate_valid((x, conditions), axis=-1)

        parameters = self.subnet(inputs, training=training, **filter_kwargs(kwargs, self.subnet.call))
        parameters = self.output_projector(parameters)
        parameters = self.transform.split_parameters(parameters)
        parameters = self.transform.constrain_parameters(parameters)

        return parameters
