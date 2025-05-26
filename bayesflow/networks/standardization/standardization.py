from collections.abc import Sequence

import keras

from bayesflow.types import Tensor, Shape
from bayesflow.utils.serialization import serialize, deserialize, serializable
from bayesflow.utils import expand_left_as, layer_kwargs


@serializable("bayesflow.networks")
class Standardization(keras.Layer):
    def __init__(self, momentum: float = 0.95, epsilon: float = 1e-6, **kwargs):
        """
        Initializes a Standardization layer that will keep track of the running mean and
        running standard deviation across a batch of tensors.

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
        if self.built:
            return

        self.moving_mean = self.add_weight(shape=(input_shape[-1],), initializer="zeros", trainable=False)
        self.moving_std = self.add_weight(shape=(input_shape[-1],), initializer="ones", trainable=False)

    def get_config(self) -> dict:
        config = {"momentum": self.momentum, "epsilon": self.epsilon}
        return serialize(config)

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))

    def call(
        self, x: Tensor, stage: str = "inference", forward: bool = True, log_det_jac: bool = False, **kwargs
    ) -> Tensor | Sequence[Tensor]:
        """
        Apply standardization or its inverse to the input tensor, optionally compute the log det of the Jacobian.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., dim).
        stage : str, optional
            Indicates the stage of computation. If "training", the running statistics
            are updated. Default is "inference".
        forward : bool, optional
            If True, apply standardization: (x - mean) / std.
            If False, apply inverse transformation: x * std + mean and return the log-determinant
            of the Jacobian. Default is True.
        log_det_jac: bool, optional
            Whether to return the log determinant of the transformation. Default is False.

        Returns
        -------
        Tensor or Sequence[Tensor]
            If `forward` is True, returns the standardized tensor, otherwise un-standardizes.
            If `log_det_jec` is True, returns a tuple: (transformed tensor, log-determinant) otherwise just
            transformed tensor.
        """
        if stage == "training":
            self._update_moments(x)

        if forward:
            x = (x - expand_left_as(self.moving_mean, x)) / expand_left_as(self.moving_std, x)
        else:
            x = expand_left_as(self.moving_mean, x) + expand_left_as(self.moving_std, x) * x

        if log_det_jac:
            ldj = keras.ops.sum(keras.ops.log(keras.ops.abs(self.moving_std)), axis=-1)
            ldj = keras.ops.broadcast_to(ldj, keras.ops.shape(x)[:-1])
            if forward:
                ldj = -ldj
            return x, ldj

        return x

    def _update_moments(self, x: Tensor):
        mean = keras.ops.mean(x, axis=tuple(range(keras.ops.ndim(x) - 1)))
        std = keras.ops.std(x, axis=tuple(range(keras.ops.ndim(x) - 1)))
        std = keras.ops.maximum(std, self.epsilon)

        self.moving_mean.assign(self.momentum * self.moving_mean + (1.0 - self.momentum) * mean)
        self.moving_std.assign(self.momentum * self.moving_std + (1.0 - self.momentum) * std)
