from typing import Sequence, Any

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from .validators import check_posterior_prior_shapes
from .dict_utils import dicts_to_arrays


def preprocess(
    post_samples: dict[str, np.ndarray],
    prior_samples: dict[str, np.ndarray],
    names: Sequence[str] = None,
    num_col: int = None,
    num_row: int = None,
    figsize: tuple = None,
    stacked: bool = False,
) -> dict[str, Any]:
    """
    Procedural wrapper that encompasses all preprocessing steps,
    including shape-checking, parameter name generation, layout configuration,
    figure initialization, and axial collapsing for loop and plot.

    Parameters
    ----------
    post_samples      : np.ndarray of shape (num_data_sets, num_post_draws, num_params)
        The posterior draws obtained from num_data_sets
    prior_samples     : np.ndarray of shape (num_data_sets, num_params)
        The prior draws obtained for generating num_data_sets
    names             : str
        Parameter name used to initialize the figure
    num_col           : int
        Number of columns for the visualization layout
    num_row           : int
        Number of rows for the visualization layout
    figsize           : tuple, optional, default: None
        Size of the figure adjusting to the display resolution
    """

    plot_data = dicts_to_arrays(post_samples, prior_samples, names)
    check_posterior_prior_shapes(plot_data["post_samples"], plot_data["prior_samples"])

    # Configure layout
    num_row, num_col = set_layout(plot_data["num_variables"], num_row, num_col, stacked)

    # Initialize figure
    f, axes = make_figure(num_row, num_col, figsize=figsize)

    plot_data["fig"] = f
    plot_data["axes"] = axes
    plot_data["num_row"] = num_row
    plot_data["num_col"] = num_col

    return plot_data


def set_layout(num_total: int, num_row: int = None, num_col: int = None, stacked: bool = False):
    """
    Determine the number of rows and columns in diagnostics visualizations.

    Parameters
    ----------
    num_total     : int
        Total number of parameters
    num_row       : int, default = None
        Number of rows for the visualization layout
    num_col       : int, default = None
        Number of columns for the visualization layout
    stacked     : bool, default = False
        Boolean that determines whether to stack the plot or not.

    Returns
    -------
    num_row       : int
        Number of rows for the visualization layout
    num_col       : int
        Number of columns for the visualization layout
    """
    if stacked:
        num_row, num_col = 1, 1
    else:
        if num_row is None and num_col is None:
            num_row = int(np.ceil(num_total / 6))
            num_col = int(np.ceil(num_total / num_row))
        elif num_row is None and num_col is not None:
            num_row = int(np.ceil(num_total / num_col))
        elif num_row is not None and num_col is None:
            num_col = int(np.ceil(num_total / num_row))

    return num_row, num_col


def make_figure(num_row: int = None, num_col: int = None, figsize: tuple = None):
    """
    Initialize a set of figures

    Parameters
    ----------
    num_row       : int
        Number of rows in a figure
    num_col       : int
        Number of columns in a figure
    figsize       : tuple
        Size of the figure adjusting to the display resolution
        or the user's choice

    Returns
    -------
    f, ax_array
        Initialized figures
    """
    if num_row == 1 and num_col == 1:
        f, axes = plt.subplots(1, 1, figsize=figsize)
    else:
        if figsize is None:
            figsize = (int(5 * num_col), int(5 * num_row))

        f, axes = plt.subplots(num_row, num_col, figsize=figsize)
    axes = np.atleast_1d(axes)

    return f, axes


def add_x_labels(axes, num_row: int = None, num_col: int = None, xlabel: str = None, label_fontsize: int = None):
    """#TODO - Deal with sequence of labels"""
    if num_row == 1:
        bottom_row = axes
    else:
        bottom_row = axes[num_row - 1, :] if num_col > 1 else axes
    for _ax in bottom_row:
        _ax.set_xlabel(xlabel, fontsize=label_fontsize)


def add_y_labels(ax_array, num_row: int = None, ylabel: str = None, label_fontsize: int = None):
    """TODO - Deal with sequence of labels"""

    if num_row == 1:  # if there is only one row, the ax array is 1D
        ax_array[0].set_ylabel(ylabel, fontsize=label_fontsize)
    # If there is more than one row, the ax array is 2D
    else:
        for _ax in ax_array[:, 0]:
            _ax.set_ylabel(ylabel, fontsize=label_fontsize)


def add_labels(
    axes: np.ndarray,
    num_row: int = None,
    num_col: int = None,
    xlabel: list[str] | str = None,
    ylabel: list[str] | str = None,
    label_fontsize: int = None,
):
    """
    Wrapper function for configuring labels for both axes.
    """
    if xlabel is not None:
        add_x_labels(axes, num_row, num_col, xlabel, label_fontsize)
    if ylabel is not None:
        add_y_labels(axes, num_row, ylabel, label_fontsize)


def remove_unused_axes(ax_array_it, num_params: int = None):
    for ax in ax_array_it[num_params:]:
        ax.remove()


def prettify_subplots(axes: np.ndarray, num_subplots: int, tick_fontsize: int = 12):
    """TODO"""
    for ax in axes.flat:
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Remove unused axes entirely
    for _ax in axes.flat[num_subplots:]:
        _ax.remove()


def make_quadratic(ax: plt.Axes, x_data: np.ndarray, y_data: np.ndarray):
    """Utility to make a subplots quadratic in order to avoid visual illusions in, e.g., recovery plots."""

    lower = min(x_data.min(), y_data.min())
    upper = max(x_data.max(), y_data.max())
    eps = (upper - lower) * 0.1
    ax.set_xlim((lower - eps, upper + eps))
    ax.set_ylim((lower - eps, upper + eps))
    ax.plot(
        [ax.get_xlim()[0], ax.get_xlim()[1]],
        [ax.get_ylim()[0], ax.get_ylim()[1]],
        color="black",
        alpha=0.9,
        linestyle="dashed",
    )


def postprocess(*args):
    """
    Procedural wrapper for postprocessing steps, including adding labels and removing unused axes.
    """

    add_labels(args)
    remove_unused_axes(args)
