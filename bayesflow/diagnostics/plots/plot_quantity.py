from collections.abc import Sequence, Mapping

import matplotlib.pyplot as plt
import numpy as np

from bayesflow.utils.plot_utils import add_titles_and_labels, make_figure, set_layout, prettify_subplots
from bayesflow.utils.dict_utils import make_variable_array


def plot_quantity(
    values: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    figsize: Sequence[int] = None,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    color: str = "#132a70",
    xlabel: str = "Ground truth",
    ylabel: str = "",
    num_col: int = None,
    num_row: int = None,
) -> plt.Figure:
    """
    Plot a quantity as a function of a variable for each variable key.

    Parameters
    ----------
    values      : np.ndarray
        The values to plot.
    targets     : np.ndarray of shape (num_datasets, num_params)
        The prior draws (true parameters) used for generating the num_datasets
    variable_keys       : list or None, optional, default: None
       Select keys from the dictionaries provided in estimates and targets.
       By default, select all keys.
    variable_names    : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
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
    num_row           : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    num_col           : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation from the expected shapes of ``estimates`` and ``targets``.
    """

    # Gather plot data and metadata into a dictionary
    values = make_variable_array(
        values,
        variable_keys=variable_keys,
        variable_names=variable_names,
    )
    variable_names = values.variable_names
    variable_keys = values.variable_keys
    targets = make_variable_array(
        targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
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

        ax.scatter(targets[:, i], values[:, i], color=color, alpha=0.5)

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
