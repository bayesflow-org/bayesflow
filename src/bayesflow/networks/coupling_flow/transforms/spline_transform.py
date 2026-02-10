import keras
import numpy as np

from bayesflow.types import Tensor
from bayesflow.utils import pad, searchsorted
from bayesflow.utils.keras_utils import shifted_softplus
from bayesflow.utils.serialization import serializable

from ._rational_quadratic import _rational_quadratic_spline
from .transform import Transform


@serializable("bayesflow.networks")
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

        if bins <= 0:
            raise ValueError("Number of bins must be strictly positive.")

        self.bins = bins
        self.min_width = max(min_width, bins * min_bin_width)
        self.min_height = max(min_height, bins * min_bin_height)
        self.min_bin_width = min_bin_width
        self.min_bin_height = min_bin_height
        self.method = method

        if self.method != "rational_quadratic":
            raise NotImplementedError("Currently, only 'rational_quadratic' spline method is supported.")

        self.method_fn = _rational_quadratic_spline

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
        batch_shape = list(keras.ops.shape(parameters)[:-1])
        parameters = keras.ops.reshape(parameters, batch_shape + [-1, self.params_per_dim])
        indices = np.cumsum(list(self.parameter_sizes.values())).tolist()
        parameters = keras.ops.split(parameters, indices, axis=-1)
        parameters = dict(zip(self.parameter_sizes.keys(), parameters))

        return parameters

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

        bin_widths = keras.ops.softmax(parameters["bin_widths"], axis=-1)
        bin_widths = (total_width - self.bins * self.min_bin_width) * bin_widths + self.min_bin_width
        bin_heights = keras.ops.softmax(parameters["bin_heights"], axis=-1)
        bin_heights = (total_height - self.bins * self.min_bin_height) * bin_heights + self.min_bin_height

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
            "affine_scale": keras.ops.squeeze(affine_scale, axis=-1),
            "affine_shift": keras.ops.squeeze(affine_shift, axis=-1),
        }

        return constrained_parameters

    def _forward(self, x: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        # avoid side effects for mutable args
        parameters = parameters.copy()

        # affine transform for outside
        scale = parameters.pop("affine_scale")
        shift = parameters.pop("affine_shift")
        affine = scale * x + shift
        affine_log_jac = keras.ops.broadcast_to(keras.ops.log(scale), keras.ops.shape(affine))

        # spline transform for inside
        bins = searchsorted(parameters["horizontal_edges"], keras.ops.expand_dims(x, axis=-1))
        bins = keras.ops.squeeze(bins, axis=-1)
        inside = (bins > 0) & (bins <= self.bins)

        upper = bins
        lower = upper - 1

        # we need to mask out invalid bins to be backend-agnostic
        # this does not matter since we will overwrite these values with the affine values anyway
        upper = keras.ops.where(inside, upper, keras.ops.ones_like(upper))
        lower = keras.ops.where(inside, lower, keras.ops.zeros_like(lower))

        # need to expand the dimensions to match the shape of the parameters for take_along_axis
        upper = keras.ops.expand_dims(upper, axis=-1)
        lower = keras.ops.expand_dims(lower, axis=-1)

        edges = {
            "left": keras.ops.take_along_axis(parameters["horizontal_edges"], lower, axis=-1),
            "right": keras.ops.take_along_axis(parameters["horizontal_edges"], upper, axis=-1),
            "bottom": keras.ops.take_along_axis(parameters["vertical_edges"], lower, axis=-1),
            "top": keras.ops.take_along_axis(parameters["vertical_edges"], upper, axis=-1),
        }
        edges = {key: keras.ops.squeeze(value, axis=-1) for key, value in edges.items()}

        derivatives = {
            "left": keras.ops.take_along_axis(parameters["derivatives"], lower, axis=-1),
            "right": keras.ops.take_along_axis(parameters["derivatives"], upper, axis=-1),
        }
        derivatives = {key: keras.ops.squeeze(value, axis=-1) for key, value in derivatives.items()}

        parameters = {"edges": edges, "derivatives": derivatives}

        # compute the spline and jacobian
        spline, spline_log_jac = self.method_fn(x, **parameters)

        z = keras.ops.where(inside, spline, affine)
        log_jac = keras.ops.where(inside, spline_log_jac, affine_log_jac)

        log_det = keras.ops.sum(log_jac, axis=-1)

        return z, log_det

    def _inverse(self, z: Tensor, parameters: dict[str, Tensor]) -> (Tensor, Tensor):
        # avoid side effects for mutable args
        parameters = parameters.copy()

        # affine transform for outside
        scale = parameters.pop("affine_scale")
        shift = parameters.pop("affine_shift")
        affine = (z - shift) / scale
        affine_log_jac = keras.ops.broadcast_to(-keras.ops.log(scale), keras.ops.shape(affine))

        # spline transform for inside
        bins = searchsorted(parameters["vertical_edges"], keras.ops.expand_dims(z, axis=-1))
        bins = keras.ops.squeeze(bins, axis=-1)
        inside = (bins > 0) & (bins <= self.bins)

        upper = bins
        lower = upper - 1

        # we need to mask out invalid bins to be backend-agnostic
        # this does not matter since we will overwrite these values with the affine values anyway
        upper = keras.ops.where(inside, upper, keras.ops.ones_like(upper))
        lower = keras.ops.where(inside, lower, keras.ops.zeros_like(lower))

        # need to expand the dimensions to match the shape of the parameters for take_along_axis
        upper = keras.ops.expand_dims(upper, axis=-1)
        lower = keras.ops.expand_dims(lower, axis=-1)

        edges = {
            "left": keras.ops.take_along_axis(parameters["horizontal_edges"], lower, axis=-1),
            "right": keras.ops.take_along_axis(parameters["horizontal_edges"], upper, axis=-1),
            "bottom": keras.ops.take_along_axis(parameters["vertical_edges"], lower, axis=-1),
            "top": keras.ops.take_along_axis(parameters["vertical_edges"], upper, axis=-1),
        }
        edges = {key: keras.ops.squeeze(value, axis=-1) for key, value in edges.items()}

        derivatives = {
            "left": keras.ops.take_along_axis(parameters["derivatives"], lower, axis=-1),
            "right": keras.ops.take_along_axis(parameters["derivatives"], upper, axis=-1),
        }
        derivatives = {key: keras.ops.squeeze(value, axis=-1) for key, value in derivatives.items()}

        parameters = {"edges": edges, "derivatives": derivatives}

        # compute the spline and jacobian
        spline, spline_log_jac = self.method_fn(z, **parameters, inverse=True)

        x = keras.ops.where(inside, spline, affine)
        log_jac = keras.ops.where(inside, spline_log_jac, affine_log_jac)

        log_det = keras.ops.sum(log_jac, axis=-1)

        return x, log_det
