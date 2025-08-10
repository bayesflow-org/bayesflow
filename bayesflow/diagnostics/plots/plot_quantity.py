from collections.abc import Callable, Sequence, Mapping

import matplotlib.pyplot as plt
import numpy as np

from bayesflow.utils.dict_utils import make_variable_array, dicts_to_arrays, filter_kwargs, compute_test_quantities
from bayesflow.utils.plot_utils import (
    add_titles_and_labels,
    make_figure,
    set_layout,
    prettify_subplots,
)
from bayesflow.utils.validators import check_estimates_prior_shapes


def plot_quantity(
    values: Mapping[str, np.ndarray] | np.ndarray | Callable,
    *,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    estimates: Mapping[str, np.ndarray] | np.ndarray | None = None,
    test_quantities: dict[str, Callable] = None,
    figsize: Sequence[int] = None,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    color: str = "#132a70",
    s: float = 25.0,
    marker: str = "o",
    alpha: float = 0.5,
    xlabel: str = "Ground truth",
    ylabel: str = "",
    num_col: int = None,
    num_row: int = None,
    default_name: str = "v",
) -> plt.Figure:
    """
    Plot a quantity as a function of a variable for each variable key.

    The function supports the following different combinations to pass
    or compute the values:

    1. pass `values` as an array of shape (num_datasets,) or (num_datasets, num_variables)
    2. pass `values` as a dictionary with the keys 'values', 'metric_name' and 'variable_names'
       as provided by the metrics functions. Note that the functions have to be called
       without aggregation to obtain value per dataset.
    3. pass a function to `values`, as well as `estimates`. The function should have the
       signature fn(estimates, targets, [aggregation]) and return an object like the
       `values` described in the previous options.

    Parameters
    ----------
    values      : dict[str, np.ndarray] | np.ndarray | Callable,
        The value of the quantity to plot. One of the following:

        1. an array of shape (num_datasets,) or (num_datasets, num_variables)
        2. a dictionary with the keys 'values', 'metric_name' and 'variable_names'
           as provided by the metrics functions. Note that the functions have to be called
           without aggregation to obtain value per dataset.
        3. a callable, requires passing `estimates` as well. The function should have the
           signature fn(estimates, targets, [aggregation]) and return an object like the
           ones described in the previous options.
    targets     : dict[str, np.ndarray] | np.ndarray,
        The parameter values plotted on the axis.
    variable_keys       : list or None, optional, default: None
       Select keys from the dictionary provided in samples.
       By default, select all keys.
    variable_names    : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    estimates      : np.ndarray of shape (n_data_sets, n_post_draws, n_params), optional, default:  None
        The posterior draws obtained from n_data_sets. Can only be supplied if
        `values` is of type Callable.
    test_quantities   : dict or None, optional, default: None
        A dict that maps plot titles to functions that compute
        test quantities based on estimate/target draws.

        The dict keys are automatically added to ``variable_keys``
        and ``variable_names``.
        Test quantity functions are expected to accept a dict of draws with
        shape ``(batch_size, ...)`` as the first (typically only)
        positional argument and return an NumPy array of shape
        ``(batch_size,)``.
        The functions do not have to deal with an additional
        sample dimension, as appropriate reshaping is done internally.
    figsize           : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 16
        The font size of the y-label text
    title_fontsize    : int, optional, default: 18
        The font size of the title text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and error bars
    s                 : float, optional, default: 25.0
        The marker size in points**2 for the scatter plot.
    marker            : str, optional, default: 'o'
        The marker for the scatter plot.
    alpha             : float, default: 0.5
        The opacity for the scatter plot
    num_row           : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    num_col           : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.
    default_name      : str, optional (default = "v")
        The default name to use for estimates if None provided

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation from the expected shapes of ``estimates`` and ``targets``.
    """

    if isinstance(values, Callable) and estimates is None:
        raise ValueError("Supplied a callable as `values`, but not `estimates`.")

    d = _prepare_values(
        values=values,
        targets=targets,
        estimates=estimates,
        variable_keys=variable_keys,
        variable_names=variable_names,
        test_quantities=test_quantities,
        label=None,
        default_name=default_name,
    )
    (values, targets, variable_keys, variable_names, test_quantities, _) = (
        d["values"],
        d["targets"],
        d["variable_keys"],
        d["variable_names"],
        d["test_quantities"],
        d["label"],
    )

    # store variable information at the top level for easy access
    num_variables = len(variable_names)

    # Configure layout
    num_row, num_col = set_layout(num_variables, num_row, num_col)

    # Initialize figure
    fig, axes = make_figure(num_row, num_col, figsize=figsize)

    # Loop and plot
    for i, ax in enumerate(axes.flat):
        if i >= num_variables:
            break

        ax.scatter(targets[:, i], values[:, i], color=color, alpha=alpha, s=s, marker=marker)

    prettify_subplots(axes, num_subplots=num_variables, tick_fontsize=tick_fontsize)

    # Add labels, titles, and set font sizes
    add_titles_and_labels(
        axes=axes,
        num_row=num_row,
        num_col=num_col,
        title=variable_names,
        xlabel=xlabel,
        ylabel=ylabel,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )

    fig.tight_layout()
    return fig


def _prepare_values(
    *,
    values: Mapping[str, np.ndarray] | np.ndarray | Callable,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    estimates: Mapping[str, np.ndarray] | np.ndarray | None,
    variable_keys: Sequence[str],
    variable_names: Sequence[str],
    test_quantities: dict[str, Callable],
    label: str | None,
    default_name: str,
):
    """
    Provate helper function to compute/extract the values required for plotting
    a quantity.

    Refer to pairs_quantity and plot_quantity for details.
    """
    is_values_callable = isinstance(values, Callable)
    # Optionally, compute and prepend test quantities from draws
    if test_quantities is not None:
        updated_data = compute_test_quantities(
            targets=targets,
            estimates=estimates,
            variable_keys=variable_keys,
            variable_names=variable_names,
            test_quantities=test_quantities,
        )
        variable_names = updated_data["variable_names"]
        variable_keys = updated_data["variable_keys"]
        estimates = updated_data["estimates"]
        targets = updated_data["targets"]

    if estimates is not None:
        if is_values_callable:
            values = values(estimates=estimates, targets=targets, **filter_kwargs({"aggregation": None}, values))

        data = dicts_to_arrays(
            estimates=estimates,
            targets=targets,
            variable_keys=variable_keys,
            variable_names=variable_names,
            default_name=default_name,
        )
        check_estimates_prior_shapes(data["estimates"], data["targets"])
        estimates = data["estimates"]
        targets = data["targets"]

        variable_keys = variable_keys or estimates.variable_keys
        if test_quantities is None:
            variable_names = variable_names or estimates.variable_names

    if all([key in values for key in ["values", "metric_name", "variable_names"]]):
        # output of a metric function
        label = values["metric_name"] if label is None else label
        variable_names = variable_names or values["variable_names"]
        values = values["values"]

    if hasattr(values, "variable_keys"):
        variable_keys = variable_keys or values.variable_keys
    if hasattr(values, "variable_names") and test_quantities is None:
        variable_names = variable_names or values.variable_names

    try:
        targets = make_variable_array(
            targets,
            variable_keys=variable_keys,
            variable_names=variable_names,
            default_name=default_name,
        )
    except ValueError:
        if test_quantities is not None and not is_values_callable:
            raise ValueError(
                "`test_quantities` requires specifying `values` as callable and passing `estimates "
                "to enable the computation of the values for each test quantity."
            )
        raise ValueError(
            "Length of 'variable_names' and number of variables do not match. "
            "Did you forget to specify `variable_keys`?"
        )

    variable_names = targets.variable_names
    variable_keys = targets.variable_keys

    if values.ndim == 1:
        values = values[:, None].repeat(len(variable_names), axis=-1)

    try:
        values = make_variable_array(
            values,
            variable_keys=variable_keys,
            variable_names=variable_names,
            default_name=default_name,
        )
    except ValueError:
        if test_quantities is not None and not is_values_callable:
            raise ValueError(
                "`test_quantities` requires specifying `values` as callable and passing `estimates "
                "to enable the computation of the values for each test quantity."
            )
        raise ValueError(
            "Length of 'variable_names' and number of variables do not match. "
            "Did you forget to specify `variable_keys`?"
        )

    return {
        "values": values,
        "targets": targets,
        "variable_keys": variable_keys,
        "variable_names": variable_names,
        "test_quantities": test_quantities,
        "label": label,
    }
