import keras
from keras.saving import (
    register_keras_serializable as serializable,
)

from bayesflow.types import Tensor
from bayesflow.utils import pad, searchsorted
from bayesflow.utils.keras_utils import shifted_softplus
from ._rational_quadratic import _rational_quadratic_spline
from .transform import Transform


@serializable(package="networks.coupling_flow")
class SplineTransform(Transform):
    def __init__(
        self,
        bins: int = 16,
        default_domain: (float, float, float, float) = (-5.0, 5.0, -5.0, 5.0),
        method: str = "rational_quadratic",
    ):
        super().__init__()
        self.bins = bins
        self.method = method

        if self.method != "rational_quadratic":
            raise NotImplementedError("Currently, only 'rational_quadratic' spline method is supported.")

        self.parameter_sizes = {
            "left_edge": 1,
            "bottom_edge": 1,
            "bin_widths": self.bins,
            "bin_heights": self.bins,
            "derivatives": self.bins - 1,
        }

        if default_domain[1] <= default_domain[0] or default_domain[3] <= default_domain[2]:
            raise ValueError("Invalid default domain. Must be (left, right, bottom, top).")

        self.default_left = default_domain[0]
        self.default_bottom = default_domain[2]
        self.default_bin_width = (default_domain[1] - default_domain[0]) / self.bins
        self.default_bin_height = (default_domain[3] - default_domain[2]) / self.bins

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
        bin_widths = self.default_bin_width * shifted_softplus(parameters["bin_widths"])
        bin_heights = self.default_bin_height * shifted_softplus(parameters["bin_heights"])

        total_width = keras.ops.sum(bin_widths, axis=-1, keepdims=True)
        total_height = keras.ops.sum(bin_heights, axis=-1, keepdims=True)

        affine_scale = total_height / total_width
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

        inside = (bins > 0) & (bins <= self.bins)
        inside_indices = keras.ops.stack(keras.ops.nonzero(inside), axis=-1)

        # first compute affine transform on everything
        scale = parameters["affine_scale"]
        shift = parameters["affine_shift"]
        z = scale * x + shift
        log_jac = keras.ops.broadcast_to(keras.ops.log(scale), keras.ops.shape(z))

        # overwrite inside part with spline
        upper = bins[inside]
        lower = upper - 1

        edges = {
            "left": keras.ops.take_along_axis(parameters["horizontal_edges"], lower, axis=None),
            "right": keras.ops.take_along_axis(parameters["horizontal_edges"], upper, axis=None),
            "bottom": keras.ops.take_along_axis(parameters["vertical_edges"], lower, axis=None),
            "top": keras.ops.take_along_axis(parameters["vertical_edges"], upper, axis=None),
        }
        derivatives = {
            "left": keras.ops.take_along_axis(parameters["derivatives"], lower, axis=None),
            "right": keras.ops.take_along_axis(parameters["derivatives"], upper, axis=None),
        }
        spline, jac = _rational_quadratic_spline(x[inside], edges=edges, derivatives=derivatives)
        z = keras.ops.scatter_update(z, inside_indices, spline)
        log_jac = keras.ops.scatter_update(log_jac, inside_indices, jac)

        log_det = keras.ops.sum(log_jac, axis=-1)

        return z, log_det

    def _inverse(self, z: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        bins = searchsorted(parameters["vertical_edges"], z)

        inside = (bins > 0) & (bins <= self.bins)
        inside_indices = keras.ops.stack(keras.ops.nonzero(inside), axis=-1)

        # first compute affine transform on everything
        scale = parameters["affine_scale"]
        shift = parameters["affine_shift"]
        x = (z - shift) / scale
        log_jac = keras.ops.broadcast_to(-keras.ops.log(scale), keras.ops.shape(x))

        # overwrite inside part with spline

        upper = bins[inside]
        lower = upper - 1

        edges = {
            "left": keras.ops.take_along_axis(parameters["vertical_edges"], lower, axis=None),
            "right": keras.ops.take_along_axis(parameters["vertical_edges"], upper, axis=None),
            "bottom": keras.ops.take_along_axis(parameters["horizontal_edges"], lower, axis=None),
            "top": keras.ops.take_along_axis(parameters["horizontal_edges"], upper, axis=None),
        }
        derivatives = {
            "left": keras.ops.take_along_axis(parameters["derivatives"], lower, axis=None),
            "right": keras.ops.take_along_axis(parameters["derivatives"], upper, axis=None),
        }
        spline, jac = _rational_quadratic_spline(z[inside], edges=edges, derivatives=derivatives, inverse=True)
        x = keras.ops.scatter_update(x, inside_indices, spline)
        log_jac = keras.ops.scatter_update(log_jac, inside_indices, jac)

        log_det = keras.ops.sum(log_jac, axis=-1)

        return x, log_det
