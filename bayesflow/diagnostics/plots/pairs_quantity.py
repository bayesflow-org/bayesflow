from collections.abc import Callable, Sequence, Mapping

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns


from .plot_quantity import _prepare_values


def pairs_quantity(
    values: Mapping[str, np.ndarray] | np.ndarray | Callable,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    *,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    estimates: Mapping[str, np.ndarray] | np.ndarray | None = None,
    test_quantities: dict[str, Callable] = None,
    height: float = 2.5,
    cmap: str | matplotlib.colors.Colormap = "viridis",
    alpha: float = 0.9,
    markersize: float = 8.0,
    marker: str = "o",
    label: str = None,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    colorbar_label_fontsize: int = 14,
    colorbar_tick_fontsize: int = 12,
    colorbar_width: float = 1.8,
    colorbar_height: float = 0.06,
    colorbar_offset: float = 0.06,
    vmin: float = None,
    vmax: float = None,
    default_name: str = "v",
    **kwargs,
) -> sns.PairGrid:
    """
    A pair plot function to plot quantities against their generating
    parameter values.

    The value is indicated by a colormap. The marginal distribution for
    each parameter is plotted on the diagonal. Each column displays the
    values of corresponding to the parameter in the column.

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
        Can only be supplied if `values` is a function.

        The dict keys are automatically added to ``variable_keys``
        and ``variable_names``.
        Test quantity functions are expected to accept a dict of draws with
        shape ``(batch_size, ...)`` as the first (typically only)
        positional argument and return an NumPy array of shape
        ``(batch_size,)``.
        The functions do not have to deal with an additional
        sample dimension, as appropriate reshaping is done internally.
    height      : float, optional, default: 2.5
        The height of the pair plot
    cmap       : str or Colormap, default: "viridis"
        The colormap for the plot.
    alpha       : float in [0, 1], optional, default: 0.9
        The opacity of the plot
    markersize        : float, optional, default: 8.0
        The marker size in points**2 for the scatter plot.
    marker            : str, optional, default: 'o'
        The marker for the scatter plot.
    label       : str, optional, default: None
        Label for the dataset to plot.
    label_fontsize    : int, optional, default: 14
        The font size of the x and y-label texts (parameter names)
    tick_fontsize     : int, optional, default: 12
        The font size of the axis tick labels
    colorbar_label_fontsize : int, optional, default: 14
        The font size of the colorbar label
    colorbar_tick_fontsize : int, optional, default: 12
        The font size of the colorbar tick labels
    colorbar_width : float, optional, default: 1.8
        The width of the colorbar in inches
    colorbar_height : float, optional, default: 0.06
        The height of the colorbar in inches
    colorbar_offset : float, optional, default: 0.06
        The vertical offset of the colorbar in inches
    vmin : float, optional, default: None
        Minimum value for the colormap. If None, the minimum value is
        determined from `values`.
    vmax : float, optional, default: None
        Maximum value for the colormap. If None, the maximum value is
        determined from `values`.
    default_name      : str, optional (default = "v")
        The default name to use for estimates if None provided
    **kwargs    : dict, optional
        Additional keyword arguments passed to the sns.PairGrid constructor

    Returns
    -------
    plt.Figure
        The figure instance

    Raises
    ------
    ValueError
        If a callable is supplied as `values`, but `estimates` is None.
    """

    if isinstance(values, Callable) and estimates is None:
        raise ValueError("Supplied a callable as `values`, but no `estimates`.")
    if not isinstance(values, Callable) and test_quantities is not None:
        raise ValueError(
            "Supplied `test_quantities`, but `values` is not a function. "
            "As the values have to be calculated for the test quantities, "
            "passing a function is required."
        )

    d = _prepare_values(
        values=values,
        targets=targets,
        estimates=estimates,
        variable_keys=variable_keys,
        variable_names=variable_names,
        test_quantities=test_quantities,
        label=label,
        default_name=default_name,
    )
    (values, targets, variable_keys, variable_names, test_quantities, label) = (
        d["values"],
        d["targets"],
        d["variable_keys"],
        d["variable_names"],
        d["test_quantities"],
        d["label"],
    )

    # Convert samples to pd.DataFrame
    data_to_plot = pd.DataFrame(targets, columns=variable_names)

    # initialize plot
    g = sns.PairGrid(
        data_to_plot,
        height=height,
        vars=variable_names,
        **kwargs,
    )

    vmin = values.min() if vmin is None else vmin
    vmax = values.max() if vmax is None else vmax

    # Generate grids
    dim = g.axes.shape[0]
    for i in range(dim):
        for j in range(dim):
            # if one value for each variable is supplied, use it for the corresponding column
            row_values = values[:, j] if values.ndim == 2 else values

            if i == j:
                ax = g.axes[i, j].twinx()
                ax.scatter(
                    targets[:, i],
                    values[:, i],
                    c=row_values,
                    cmap=cmap,
                    s=markersize,
                    marker=marker,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=alpha,
                )
                ax.spines["left"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
                ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)
                ax.set_ylim(vmin, vmax)

                if i > 0:
                    g.axes[i, j].get_yaxis().set_visible(False)
                    g.axes[i, j].spines["left"].set_visible(False)
                if i == dim - 1:
                    ax.set_ylabel(label, size=label_fontsize)
            else:
                g.axes[i, j].grid(alpha=0.5)
                g.axes[i, j].set_axisbelow(True)
                g.axes[i, j].scatter(
                    targets[:, j],
                    targets[:, i],
                    c=row_values,
                    cmap=cmap,
                    s=markersize,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=alpha,
                    marker=marker,
                )

    def inches_to_figure(fig, values):
        return fig.transFigure.inverted().transform(fig.dpi_scale_trans.transform(values))

    # position and draw colorbar
    _, yoffset = inches_to_figure(g.figure, [0, colorbar_offset])
    cwidth, cheight = inches_to_figure(g.figure, [colorbar_width, colorbar_offset])
    cax = g.figure.add_axes([0.5 - cwidth / 2, -yoffset - cheight, cwidth, cheight])

    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(
        matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        location="bottom",
        label=label,
        alpha=alpha,
    )

    cbar.set_label(label, size=colorbar_label_fontsize)
    cax.tick_params(labelsize=colorbar_tick_fontsize)

    dim = g.axes.shape[0]
    for i in range(dim):
        # Modify tick sizes
        for j in range(i + 1):
            g.axes[i, j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
            g.axes[i, j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)

        # adjust the font size of labels
        # the labels themselves remain the same as before, i.e., variable_names
        g.axes[i, 0].set_ylabel(variable_names[i], fontsize=label_fontsize)
        g.axes[dim - 1, i].set_xlabel(variable_names[i], fontsize=label_fontsize)

    return g
