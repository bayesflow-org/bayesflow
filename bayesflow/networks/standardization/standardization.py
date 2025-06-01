from collections.abc import Sequence

import keras

from bayesflow.types import Tensor, Shape
from bayesflow.utils.serialization import serialize, deserialize, serializable
from bayesflow.utils import expand_left_as, layer_kwargs
from bayesflow.utils.tree import flatten_shape


@serializable("bayesflow.networks")
class Standardization(keras.Layer):
    def __init__(self, momentum: float = 0.95, epsilon: float = 1e-6, **kwargs):
        """
        Initializes a Standardization layer that will keep track of the running mean and
        running standard deviation across a batch of potentially nested tensors.

        Parameters
        ----------
        momentum : float, optional
            Momentum for the exponential moving average used to update the mean and
            standard deviation during training. Must be between 0 and 1.
            Default is 0.95.
        epsilon: float, optional
            Stability parameter to avoid division by zero.
        """
        super().__init__(**layer_kwargs(kwargs))

        self.momentum = momentum
        self.epsilon = epsilon
        self.moving_mean = None
        self.moving_std = None

    def build(self, input_shape: Shape):
        flattened_shapes = flatten_shape(input_shape)
        self.moving_mean = [
            self.add_weight(shape=(shape[-1],), initializer="zeros", trainable=False) for shape in flattened_shapes
        ]
        self.moving_std = [
            self.add_weight(shape=(shape[-1],), initializer="ones", trainable=False) for shape in flattened_shapes
        ]

    def get_config(self) -> dict:
        base_config = super().get_config()
        config = {"momentum": self.momentum, "epsilon": self.epsilon}
        return base_config | serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def call(
        self,
        x: Tensor | dict[str, Tensor],
        stage: str = "inference",
        forward: bool = True,
        log_det_jac: bool = False,
        **kwargs,
    ) -> Tensor | Sequence[Tensor]:
        """
        Apply standardization or its inverse to the input tensor. Optionally compute the log determinant
        of the Jacobian (useful for flows or density estimation).

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., dim).
        stage : str, optional
            Indicates the stage of computation. If "training", running statistics are updated.
        forward : bool, optional
            If True, apply standardization: (x - mean) / std. Otherwise, inverse transform.
        log_det_jac : bool, optional
            Whether to return the log determinant of the Jacobian. Default is False.

        Returns
        -------
        Tensor or Sequence[Tensor]
            Transformed tensor, and optionally the log-determinant if `log_det_jac=True`.
        """
        flattened = keras.tree.flatten(x)
        outputs, log_det_jacs = [], []

        for i, val in enumerate(flattened):
            if stage == "training":
                self._update_moments(val, i)

            mean = expand_left_as(self.moving_mean[i], val)
            std = expand_left_as(self.moving_std[i], val)

            if forward:
                out = (val - mean) / std
            else:
                out = mean + std * val

            outputs.append(out)

            if log_det_jac:
                ldj = keras.ops.sum(keras.ops.log(keras.ops.abs(std)), axis=-1)
                # For convenience, tile to batch shape of val
                ldj = keras.ops.tile(ldj, keras.ops.shape(val)[:-1])
                log_det_jacs.append(-ldj if forward else ldj)

        outputs = keras.tree.pack_sequence_as(x, outputs)
        if log_det_jac:
            log_det_jacs = keras.tree.pack_sequence_as(x, log_det_jacs)
            return outputs, log_det_jacs

        return outputs

    def _update_moments(self, x: Tensor, index: int):
        mean = keras.ops.mean(x, axis=tuple(range(keras.ops.ndim(x) - 1)))
        std = keras.ops.std(x, axis=tuple(range(keras.ops.ndim(x) - 1)))
        std = keras.ops.maximum(std, self.epsilon)

        self.moving_mean[index].assign(self.momentum * self.moving_mean[index] + (1.0 - self.momentum) * mean)
        self.moving_std[index].assign(self.momentum * self.moving_std[index] + (1.0 - self.momentum) * std)
