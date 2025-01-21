import keras
import numpy as np
from keras.saving import (
    register_keras_serializable as serializable,
)

from bayesflow.types import Tensor
from bayesflow.utils import expand_as, pad, searchsorted
from bayesflow.utils.keras_utils import shifted_softplus
from ._rational_quadratic import _rational_quadratic_spline
from .transform import Transform


@serializable(package="networks.coupling_flow")
class SplineTransform(Transform):
    def __init__(
        self,
        bins: int = 16,
        default_domain: (float, float, float, float) = (-3.0, 3.0, -3.0, 3.0),
        min_width: float = 1.0,
        min_height: float = 1.0,
        min_bin_width: float = 0.1,
        min_bin_height: float = 0.1,
        method: str = "rational_quadratic",
    ):
        super().__init__()
        self.bins = bins
        self.min_width = max(min_width, bins * min_bin_width)
        self.min_height = max(min_height, bins * min_bin_height)
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.method = method

        if self.method != "rational_quadratic":
            raise NotImplementedError("Currently, only 'rational_quadratic' spline method is supported.")

        # we slightly over-parametrize to allow for better constraints
        # this may also improve convergence due to redundancy
        self.parameter_sizes = {
            "left_edge": 1,
            "bottom_edge": 1,
            "total_width": 1,
            "total_height": 1,
            "bin_widths": self.bins,
            "bin_heights": self.bins,
            "derivatives": self.bins - 1,
        }

        if default_domain[1] <= default_domain[0] or default_domain[3] <= default_domain[2]:
            raise ValueError("Invalid default domain. Must be (left, right, bottom, top).")

        self.default_left = default_domain[0]
        self.default_bottom = default_domain[2]
        self.default_width = default_domain[1] - default_domain[0]
        self.default_height = default_domain[3] - default_domain[2]

        if self.default_width < self.min_width:
            raise ValueError(f"Default width must be greater than minimum width ({self.min_width}).")

        if self.default_height < self.min_height:
            raise ValueError(f"Default height must be greater than minimum height ({self.min_height}).")

        self._shift = np.sinh(1.0) * np.log(np.e - 1.0)

    @property
    def params_per_dim(self) -> int:
        return sum(self.parameter_sizes.values())

    def split_parameters(self, parameters: Tensor) -> dict[str, Tensor]:
        p = {}

        start = 0
        for key, value in self.parameter_sizes.items():
            stop = start + value
            p[key] = keras.ops.take(parameters, indices=list(range(start, stop)), axis=-1)
            start = stop

        return p

    def constrain_parameters(self, parameters: dict[str, Tensor]) -> dict[str, Tensor]:
        left_edge = parameters["left_edge"] + self.default_left
        bottom_edge = parameters["bottom_edge"] + self.default_bottom

        # strictly positive (softplus)
        # scales logarithmically to infinity (arcsinh)
        # 1 when network outputs 0 (shift)
        total_width = keras.ops.arcsinh(keras.ops.softplus(parameters["total_width"] + self._shift))
        total_width = (self.default_width - self.min_width) * total_width + self.min_width

        total_height = keras.ops.arcsinh(keras.ops.softplus(parameters["total_height"] + self._shift))
        total_height = (self.default_height - self.min_height) * total_height + self.min_height

        bin_widths = (total_width - self.bins * self.min_bin_width) * keras.ops.softmax(
            parameters["bin_widths"], axis=-1
        ) + self.min_bin_width
        bin_heights = (total_height - self.bins * self.min_bin_height) * keras.ops.softmax(
            parameters["bin_heights"], axis=-1
        ) + self.min_bin_height

        # dy / dx
        affine_scale = total_height / total_width
        # y = a * x + b -> b = y - a * x
        affine_shift = bottom_edge - affine_scale * left_edge

        horizontal_edges = keras.ops.cumsum(bin_widths, axis=-1)
        horizontal_edges = pad(horizontal_edges, 0.0, 1, axis=-1, side="left")
        horizontal_edges = left_edge + horizontal_edges

        vertical_edges = keras.ops.cumsum(bin_heights, axis=-1)
        vertical_edges = pad(vertical_edges, 0.0, 1, axis=-1, side="left")
        vertical_edges = bottom_edge + vertical_edges

        derivatives = shifted_softplus(parameters["derivatives"])
        derivatives = pad(derivatives, affine_scale, 1, axis=-1, side="both")

        constrained_parameters = {
            "horizontal_edges": horizontal_edges,
            "vertical_edges": vertical_edges,
            "derivatives": derivatives,
            "affine_scale": affine_scale,
            "affine_shift": affine_shift,
        }

        return constrained_parameters

    def _forward(self, x: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        # x.shape == ([B, ...], D)
        # parameters.shape == ([B, ...], bins)
        bins = searchsorted(parameters["horizontal_edges"], x)

        # inside check is right-inclusive because searchsorted is right-inclusive
        inside = (bins > 0) & (bins <= self.bins)
        # inside_indices.shape == (n_inside, ndim(x))
        inside_indices = keras.ops.stack(keras.ops.nonzero(inside), axis=-1)

        # first compute affine transform on everything
        scale = parameters["affine_scale"]
        shift = parameters["affine_shift"]
        z = scale * x + shift
        log_jac = keras.ops.broadcast_to(keras.ops.log(scale), keras.ops.shape(z))

        # overwrite inside part with spline
        upper = bins[inside]
        upper = expand_as(upper, parameters["horizontal_edges"], side="right")

        lower = upper - 1

        # select batch elements that are inside
        parameters_inside = {key: value[inside_indices[:, :-1]] for key, value in parameters.items()}
        parameters_inside = {key: keras.ops.squeeze(value, axis=1) for key, value in parameters_inside.items()}

        # select bin parameters for inside elements
        edges = {
            "left": keras.ops.take_along_axis(parameters_inside["horizontal_edges"], lower, axis=-1),
            "right": keras.ops.take_along_axis(parameters_inside["horizontal_edges"], upper, axis=-1),
            "bottom": keras.ops.take_along_axis(parameters_inside["vertical_edges"], lower, axis=-1),
            "top": keras.ops.take_along_axis(parameters_inside["vertical_edges"], upper, axis=-1),
        }
        edges = {key: keras.ops.squeeze(value, axis=-1) for key, value in edges.items()}
        derivatives = {
            "left": keras.ops.take_along_axis(parameters_inside["derivatives"], lower, axis=-1),
            "right": keras.ops.take_along_axis(parameters_inside["derivatives"], upper, axis=-1),
        }
        derivatives = {key: keras.ops.squeeze(value, axis=-1) for key, value in derivatives.items()}

        # compute spline and jacobian
        spline, jac = _rational_quadratic_spline(x[inside], edges=edges, derivatives=derivatives)
        z = keras.ops.scatter_update(z, inside_indices, spline)
        log_jac = keras.ops.scatter_update(log_jac, inside_indices, jac)

        log_det = keras.ops.sum(log_jac, axis=-1)

        return z, log_det

    def _inverse(self, z: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        # z.shape == ([B, ...], D)
        # parameters.shape == ([B, ...], bins)
        bins = searchsorted(parameters["vertical_edges"], z)

        # inside check is right-inclusive because searchsorted is right-inclusive
        inside = (bins > 0) & (bins <= self.bins)
        # inside_indices.shape == (n_inside, ndim(x))
        inside_indices = keras.ops.stack(keras.ops.nonzero(inside), axis=-1)

        # first compute affine transform on everything
        scale = parameters["affine_scale"]
        shift = parameters["affine_shift"]
        x = (z - shift) / scale
        log_jac = keras.ops.broadcast_to(-keras.ops.log(scale), keras.ops.shape(x))

        # overwrite inside part with spline
        upper = bins[inside]
        upper = expand_as(upper, parameters["horizontal_edges"], side="right")

        lower = upper - 1

        # select batch elements that are inside
        parameters_inside = {key: value[inside_indices[:, :-1]] for key, value in parameters.items()}
        parameters_inside = {key: keras.ops.squeeze(value, axis=1) for key, value in parameters_inside.items()}

        # select bin parameters for inside elements
        edges = {
            "left": keras.ops.take_along_axis(parameters_inside["horizontal_edges"], lower, axis=-1),
            "right": keras.ops.take_along_axis(parameters_inside["horizontal_edges"], upper, axis=-1),
            "bottom": keras.ops.take_along_axis(parameters_inside["vertical_edges"], lower, axis=-1),
            "top": keras.ops.take_along_axis(parameters_inside["vertical_edges"], upper, axis=-1),
        }
        edges = {key: keras.ops.squeeze(value, axis=-1) for key, value in edges.items()}
        derivatives = {
            "left": keras.ops.take_along_axis(parameters_inside["derivatives"], lower, axis=-1),
            "right": keras.ops.take_along_axis(parameters_inside["derivatives"], upper, axis=-1),
        }
        derivatives = {key: keras.ops.squeeze(value, axis=-1) for key, value in derivatives.items()}

        # compute spline and jacobian
        spline, jac = _rational_quadratic_spline(z[inside], edges=edges, derivatives=derivatives, inverse=True)
        x = keras.ops.scatter_update(x, inside_indices, spline)
        log_jac = keras.ops.scatter_update(log_jac, inside_indices, jac)

        log_det = keras.ops.sum(log_jac, axis=-1)

        return x, log_det
