
import numpy as np
import matplotlib.pyplot as plt

from .exceptions.shape_error import ShapeError


def check_posterior_prior_shapes(post_samples, prior_samples):
    """
    Checks requirements for the shapes of posterior and prior draws as
    necessitated by most diagnostic functions.

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of `post_samples` and `prior_samples`.
    """

    if len(post_samples.shape) != 3:
        raise ShapeError(
            "post_samples should be a 3-dimensional array, with the "
            + "first dimension being the number of (simulated) data sets, "
            + "the second dimension being the number of posterior draws per data set, "
            + "and the third dimension being the number of parameters (marginal distributions), "
            + f"but your input has dimensions {len(post_samples.shape)}"
        )
    elif len(prior_samples.shape) != 2:
        raise ShapeError(
            "prior_samples should be a 2-dimensional array, with the "
            + "first dimension being the number of (simulated) data sets / prior draws "
            + "and the second dimension being the number of parameters (marginal distributions), "
            + f"but your input has dimensions {len(prior_samples.shape)}"
        )
    elif post_samples.shape[0] != prior_samples.shape[0]:
        raise ShapeError(
            "The number of elements over the first dimension of post_samples and prior_samples"
            + f"should match, but post_samples has {post_samples.shape[0]} and prior_samples has "
            + f"{prior_samples.shape[0]} elements, respectively."
        )
    elif post_samples.shape[-1] != prior_samples.shape[-1]:
        raise ShapeError(
            "The number of elements over the last dimension of post_samples and prior_samples"
            + f"should match, but post_samples has {post_samples.shape[1]} and prior_samples has "
            + f"{prior_samples.shape[-1]} elements, respectively."
        )


def get_count_and_names(
        samples,
        names: list = None,
        symbol: str = None,
        n_objects: int = None
):
    """
    Determine the number of objects, such as parameters or models,
    and their respective names if None given.

    Parameters
    ----------
    samples         : np.ndarray of shape(..., n_objects)
        The objects of interest
    names           : list[str], optional, default: None
        The names of individual object
    symbol          : str, optional, default: None
        The symbol used for naming the individual object.
        If none given, default is associated with a parameter named $\\theta$.
    n_objects        : int, optional, default: None
        The number of individual objects

    Returns
    -------
    n_objects       : int
        Number of individual objects
    names           : list[str]
        List of names for the individual object
    """
    if n_objects is None:
        n_objects = samples.shape[-1]
    if names is None:
        if symbol is None:
            symbol = "\\theta"
        names = [f"${symbol}_{{{i}}}$" for i in range(1, n_objects + 1)]

    return n_objects, names


def configure_layout(
        n_total: int,
        n_row: int = None,
        n_col: int = None,
        stacked: bool = False
):
    """
    Determine the number of rows and columns in diagnostics visualizations.

    Parameters
    ----------
    n_total     : int
        Total number of parameters
    n_row       : int, default = None
        Number of rows for the visualization layout
    n_col       : int, default = None
        Number of columns for the visualization layout
    stacked     : bool, default = False
        Boolean that determines whether to stack the plot or not.

    Returns
    -------
    n_row       : int
        Number of rows for the visualization layout
    n_col       : int
        Number of columns for the visualization layout
    """
    if stacked:
        n_row, n_col = 1, 1
    else:
        if n_row is None and n_col is None:
            n_row = int(np.ceil(n_total / 6))
            n_col = int(np.ceil(n_total / n_row))
        elif n_row is None and n_col is not None:
            n_row = int(np.ceil(n_total / n_col))
        elif n_row is not None and n_col is None:
            n_col = int(np.ceil(n_total / n_row))

    return n_row, n_col


def initialize_figure(
        n_row: int = None,
        n_col: int = None,
        fig_size: tuple = None,
):
    """
    Initialize a set of figures

    Parameters
    ----------
    n_row       : int
        Number of rows in a figure
    n_col       : int
        Number of columns in a figure
    stacked     : bool
        Whether subplots in a figure are stacked by rows
    fig_size    : tuple
        Size of the figure adjusting to the display resolution
        or the designer's desire

    Returns
    -------
    f, axarr
        Initialized figures
    """
    if n_row == 1 and n_col == 1:
        f, axarr = plt.subplots(1, 1, figsize=fig_size)
    else:
        if fig_size is None:
            fig_size = (int(5 * n_col), int(5 * n_row))

        f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)

    return f, axarr


def collapse_axes(axarr, n_row: int = 1, n_col: int = 1):
    """
    Collapse a 2D array of subplot Axes into a 1D array

    Parameters
    ----------
    axarr      : 2D array of Axes
        An array of axes for subplots
    n_row      : int, default: 1
        Number of rows for the axes
    n_col      : int, default: 1
        Number of columns for the axes

    Returns
    -------
    ax          : 1D array of Axes
        Collapsed axes for subplots
    """

    ax = np.atleast_1d(axarr)
    # turn axarr into 1D list
    if n_row > 1 or n_col > 1:
        ax = axarr.flat
    else:
        ax = axarr

    return ax


def add_xlabels(
        axarr,
        n_row: int = None,
        n_col: int = None,
        xlabel: str = None,
        label_fontsize: int = None
):
    # Only add x-labels to the bottom row
    bottom_row = axarr if n_row == 1 else axarr[0] if n_col == 1 else axarr[n_row - 1, :]
    for _ax in bottom_row:
        _ax.set_xlabel(xlabel, fontsize=label_fontsize)


def add_ylabels(
        axarr,
        n_row: int = None,
        ylabel: str = None,
        label_fontsize: int = None
):
    # Only add y-labels to right left-most row
    if n_row == 1:  # if there is only one row, the ax array is 1D
        axarr[0].set_ylabel(ylabel, fontsize=label_fontsize)
    # If there is more than one row, the ax array is 2D
    else:
        for _ax in axarr[:, 0]:
            _ax.set_ylabel(ylabel, fontsize=label_fontsize)


def add_labels(
        axarr,
        n_row: int = None,
        n_col: int = None,
        xlabel: str = None,
        ylabel: str = None,
        label_fontsize: int = None
):
    """
    Wrapper function for configuring labels for both axes.
    """
    add_xlabels(axarr, n_row, n_col, xlabel, label_fontsize)
    add_ylabels(axarr, n_row, ylabel, label_fontsize)


def remove_unused_axes(axarr_it, n_params: int = None):
    for _ax in axarr_it[n_params:]:
        _ax.remove()


def preprocess(
        post_samples,
        prior_samples,
        fig_size: tuple = None,
        collapse: bool = True
):
    """
    Procedural wrapper that encompasses all preprocessing steps,
    including shape-checking, parameter name generation, layout configuration,
    figure initialization, and axial collapsing for loop and plot.

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets
    fig_size          : tuple, optional, default: None
        Size of the figure adjusting to the display resolution
    stacked          : bool, optional, default: False
        Whether subplots in a figure are stacked by rows
    collapse         : bool, optional, default: True
        Whether subplots in a figure are collapsed into rows
    """

    # Sanity check
    check_posterior_prior_shapes(post_samples, prior_samples)

    # Determine parameters and parameter names
    n_params, param_names = get_count_and_names(post_samples)

    # Configure layout
    n_row, n_col = configure_layout(n_params)

    # Initialize figure
    f, axarr = initialize_figure(n_row, n_col, fig_size=fig_size)

    # turn axarr into 1D list
    if collapse:
        axarr_it = collapse_axes(axarr, n_row, n_col)
    else:
        axarr_it = axarr

    return f, axarr, axarr_it, n_row, n_col, n_params, param_names


def postprocess(*args):
    """
    Procedural wrapper for postprocessing steps, including adding labels and removing unused axes.
    """

    add_labels(args)
    remove_unused_axes(args)
