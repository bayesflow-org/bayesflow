"""
A collection of utility functions, mostly used for internal purposes.
"""

from . import (
    keras_utils,
    logging,
    numpy_utils,
    serialization,
    tree,
)

from .callbacks import detailed_loss_callback

from .devices import devices

from .dict_utils import (
    convert_args,
    convert_kwargs,
    filter_kwargs,
    layer_kwargs,
    model_kwargs,
    sequential_kwargs,
    split_tensors,
    split_arrays,
    squeeze_inner_estimates_dict,
)

from .dispatch import (
    find_network,
    find_permutation,
    find_pooling,
    find_recurrent_net,
    find_summary_network,
    find_inference_network,
    find_distribution,
)

from .ecdf import simultaneous_ecdf_bands, ranks

from .functional import batched_call

from .git import (
    issue_url,
    pull_url,
    repo_url,
)

from .hparam_utils import find_batch_size, find_memory_budget
from .integrate import integrate, integrate_stochastic

from .io import (
    pickle_load,
    format_bytes,
    parse_bytes,
)

from .jacobian import (
    jacobian,
    jacobian_trace,
    jvp,
    vjp,
)

from .optimal_transport import optimal_transport

from .plot_utils import (
    check_estimates_prior_shapes,
    prepare_plot_data,
    add_titles_and_labels,
    prettify_subplots,
    make_quadratic,
    add_metric,
)
from .serialization import serialize_value_or_type, deserialize_value_or_type

from .tensor_utils import (
    concatenate_valid,
    concatenate_valid_shapes,
    expand,
    expand_as,
    expand_to,
    expand_left,
    expand_left_as,
    expand_left_to,
    expand_right,
    expand_right_as,
    expand_right_to,
    expand_tile,
    fill_triangular_matrix,
    pad,
    positive_diag,
    searchsorted,
    size_of,
    stack_valid,
    tile_axis,
    tree_concatenate,
    tree_stack,
    weighted_mean,
)

from .classification import calibration_curve, confusion_matrix

from .validators import check_lengths_same

from ._docs import _add_imports_to_all

_add_imports_to_all(include_modules=["keras_utils", "logging", "numpy_utils", "serialization", "tree"])
