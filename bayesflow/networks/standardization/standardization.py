from collections.abc import Sequence

import keras

from bayesflow.types import Tensor, Shape
from bayesflow.utils.serialization import serializable
from bayesflow.utils import expand_left_as, layer_kwargs
from bayesflow.utils.tree import flatten_shape


@serializable("bayesflow.networks")
class Standardization(keras.Layer):
    def __init__(self, **kwargs):
        """
        Initializes a Standardization layer that will keep track of the running mean and
        running standard deviation across a batch of potentially nested tensors.

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
        self.moving_m2 = None
        self.count = None

    def moving_std(self, index: int) -> Tensor:
        """Calculates the standard deviation from the moving M^2 at the given index and the count.

        Important: Where M^2=0, it will return a standard deviation of 1 instead of 0, even if count > 0.
        """
        return keras.ops.where(
            self.moving_m2[index] > 0,
            keras.ops.sqrt(self.moving_m2[index] / self.count[index]),
            1.0,
        )

    def build(self, input_shape: Shape):
        flattened_shapes = flatten_shape(input_shape)

        self.moving_mean = [
            self.add_weight(shape=(shape[-1],), initializer="zeros", trainable=False) for shape in flattened_shapes
        ]
        self.moving_m2 = [
            self.add_weight(shape=(shape[-1],), initializer="zeros", trainable=False) for shape in flattened_shapes
        ]
        self.count = [self.add_weight(shape=(), initializer="zeros", trainable=False) for _ in flattened_shapes]

    def call(
        self,
        x: Tensor | dict[str, Tensor],
        stage: str = "inference",
        forward: bool = True,
        log_det_jac: bool = False,
        transformation_type: str = "location_scale",
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
        transformation_type: str, optional
            The type of inverse transform to apply. Only relevant if used with arbitrary point estimates.
            Default is "location_scale", i.e., undo standardization.

        Returns
        -------
        Tensor or Sequence[Tensor]
            Transformed tensor, and optionally the log-determinant if `log_det_jac=True`.
        """
        if (forward or log_det_jac) and transformation_type != "location_scale":
            raise ValueError(
                'Non-default transformation (i.e. transformation_type != "location_scale") '
                "is not supported for forward or log_det_jac."
            )

        flattened = keras.tree.flatten(x)
        outputs, log_det_jacs = [], []

        for idx, val in enumerate(flattened):
            if stage == "training":
                self._update_moments(val, idx)

            mean = expand_left_as(self.moving_mean[idx], val)
            # moving_std will return 1 in the case of std=0, so no further checks are necessary here
            std = expand_left_as(self.moving_std(idx), val)

            if forward:
                out = (val - mean) / std
            else:
                match transformation_type:
                    case "location_scale":
                        # x_i = x_i' * sigma_i + mu_i
                        out = val * std + mean
                    case "both_sides_scale":
                        # x_ij = sigma_i * x_ij' * sigma_j
                        out = val * std * keras.ops.moveaxis(std, -1, -2)
                    case "left_side_scale":
                        # x_ij = sigma_i * x_ij'
                        out = val * keras.ops.moveaxis(std, -1, -2)
                    case _:
                        out = val

            outputs.append(out)

            if log_det_jac:
                ldj = keras.ops.sum(keras.ops.log(keras.ops.abs(std)), axis=-1)
                ldj = keras.ops.tile(ldj, keras.ops.shape(val)[:-1])
                log_det_jacs.append(-ldj if forward else ldj)

        outputs = keras.tree.pack_sequence_as(x, outputs)
        if log_det_jac:
            log_det_jacs = keras.tree.pack_sequence_as(x, log_det_jacs)
            return outputs, log_det_jacs

        return outputs

    def _update_moments(self, x: Tensor, index: int):
        """
        Incrementally updates the running mean and variance (M2) per feature using a numerically
        stable online algorithm.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (..., features), where all axes except the last are treated as batch/sample axes.
            The method computes batch-wise statistics by aggregating over all non-feature axes and updates the
            running totals (mean, M2, and sample count) accordingly.
        index : int
            The index of the corresponding running statistics to be updated.
        """

        reduce_axes = tuple(range(x.ndim - 1))
        batch_count = keras.ops.cast(keras.ops.prod(keras.ops.shape(x)[:-1]), self.count[index].dtype)

        # Compute batch mean and M2 per feature
        batch_mean = keras.ops.mean(x, axis=reduce_axes)
        batch_m2 = keras.ops.sum((x - expand_left_as(batch_mean, x)) ** 2, axis=reduce_axes)

        # Read current totals
        mean = self.moving_mean[index]
        m2 = self.moving_m2[index]
        count = self.count[index]

        total_count = count + batch_count
        delta = batch_mean - mean

        new_mean = mean + delta * (batch_count / total_count)
        new_m2 = m2 + batch_m2 + (delta**2) * (count * batch_count / total_count)

        self.moving_mean[index].assign(new_mean)
        self.moving_m2[index].assign(new_m2)
        self.count[index].assign(total_count)
