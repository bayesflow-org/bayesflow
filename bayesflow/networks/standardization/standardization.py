from collections.abc import Sequence

import keras

from bayesflow.types import Tensor, Shape
from bayesflow.utils.serialization import serializable
from bayesflow.utils import expand_left_as, layer_kwargs


@serializable("bayesflow.networks")
class Standardization(keras.Layer):
    def __init__(self, **kwargs):
        """
        Initializes a Standardization layer that tracks the running mean and standard deviation per
        feature for online normalization.

        The layer computes and stores running estimates of the mean and variance using a numerically
        stable online algorithm, allowing for consistent normalization during both training and inference,
        regardless of batch composition.

        Parameters
        ----------
        **kwargs
            Additional keyword arguments passed to the base Keras Layer.

        Notes
        -----
        """
        super().__init__(**layer_kwargs(kwargs))

        self.moving_mean = None
        self.moving_M2 = None
        self.count = None

    @property
    def moving_std(self):
        return keras.ops.sqrt(self.moving_M2 / self.count)

    def build(self, input_shape: Shape):
        feature_dim = input_shape[-1]
        self.moving_mean = self.add_weight(shape=(feature_dim,), initializer="zeros", trainable=False)
        self.moving_M2 = self.add_weight(shape=(feature_dim,), initializer="ones", trainable=False)
        self.count = self.add_weight(shape=(), initializer="zeros", trainable=False, dtype="int64")

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
        """
        Incrementally updates the running mean and variance (M2) per feature using a numerically
        stable online algorithm.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., features), where all axes except the last are treated as batch/sample axes.
            The method computes batch-wise statistics by aggregating over all non-feature axes and updates the
            running totals (mean, M2, and sample count) accordingly.
        """

        reduce_axes = tuple(range(x.ndim - 1))
        batch_count = keras.ops.cast(keras.ops.shape(x)[0], self.count.dtype)

        # Compute batch mean and M2 per feature
        batch_mean = keras.ops.mean(x, axis=reduce_axes)
        batch_M2 = keras.ops.sum((x - expand_left_as(batch_mean, x)) ** 2, axis=reduce_axes)

        # Read current totals
        mean = self.moving_mean
        M2 = self.moving_M2
        count = self.count

        total_count = count + batch_count
        delta = batch_mean - mean

        new_mean = mean + delta * (batch_count / total_count)
        new_M2 = M2 + batch_M2 + (delta**2) * (count * batch_count / total_count)

        self.moving_mean.assign(new_mean)
        self.moving_M2.assign(new_M2)
        self.count.assign(total_count)
