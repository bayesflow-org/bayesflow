from typing import Sequence

import numpy as np
import pandas as pd
import seaborn as sns

from bayesflow.utils import logging
from bayesflow.utils.dict_utils import dicts_to_arrays


def pairs_samples(
    samples: dict[str, np.ndarray] | np.ndarray = None,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    height: float = 2.5,
    color: str | tuple = "#132a70",
    alpha: float = 0.9,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    **kwargs,
) -> sns.PairGrid:
    """
    A more flexible pair plot function for multiple distributions based upon
    collected samples.

    Parameters
    ----------
    samples     : dict[str, Tensor], default: None
        Sample draws from any dataset
    variable_keys       : list or None, optional, default: None
       Select keys from the dictionary provided in samples.
       By default, select all keys.
    variable_names    : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    height      : float, optional, default: 2.5
        The height of the pair plot
    color       : str, optional, default : '#8f2727'
        The color of the plot
    alpha       : float in [0, 1], optional, default: 0.9
        The opacity of the plot
    label_fontsize    : int, optional, default: 14
        The font size of the x and y-label texts (parameter names)
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    **kwargs    : dict, optional
        Additional keyword arguments passed to the sns.PairGrid constructor
    """

    plot_data = dicts_to_arrays(
        estimates=samples,
        variable_keys=variable_keys,
        variable_names=variable_names,
    )

    g = _pairs_samples(
        plot_data=plot_data,
        height=height,
        color=color,
        alpha=alpha,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
    )

    return g


def _pairs_samples(
    plot_data: dict,
    height: float = 2.5,
    color: str | tuple = "#132a70",
    alpha: float = 0.9,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    **kwargs,
) -> sns.PairGrid:
    # internal version of pairs_samples creating the seaborn plot

    # Parameters
    # ----------
    # plot_data   : output of bayesflow.utils.dict_utils.dicts_to_arrays
    # other arguments are documented in pairs_samples

    estimates_shape = plot_data["estimates"].shape
    if len(estimates_shape) != 2:
        raise ValueError(
            f"Samples for a single distribution should be a matrix, but "
            f"your samples array has a shape of {estimates_shape}."
        )

    # Convert samples to pd.DataFrame
    data_to_plot = pd.DataFrame(plot_data["estimates"], columns=plot_data["variable_names"])

    # initialize plot
    artist = sns.PairGrid(data_to_plot, height=height, **kwargs)

    # Generate grids
    # in the off diagonal plots, the grids appears in front of the points/densities
    # TODO: can we put the grid in the background somehow?
    dim = artist.axes.shape[0]
    for i in range(dim):
        for j in range(dim):
            artist.axes[i, j].grid(alpha=0.5)

    # add histograms + KDEs to the diagonal
    artist.map_diag(sns.histplot, fill=True, color=color, alpha=alpha, kde=True)

    # Incorporate exceptions for generating KDE plots
    try:
        artist.map_lower(sns.kdeplot, fill=True, color=color, alpha=alpha)
    except Exception as e:
        logging.exception("KDE failed due to the following exception:\n" + repr(e) + "\nSubstituting scatter plot.")
        artist.map_lower(sns.scatterplot, alpha=0.6, s=40, edgecolor="k", color=color, lw=0)

    artist.map_upper(sns.scatterplot, alpha=0.6, s=40, edgecolor="k", color=color, lw=0)

    dim = artist.axes.shape[0]
    for i in range(dim):
        # Modify tick sizes
        for j in range(i + 1):
            artist.axes[i, j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
            artist.axes[i, j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)

        # adjust font size of labels
        # the labels themselves remain the same as before, i.e., variable_names
        artist.axes[i, 0].set_ylabel(plot_data["variable_names"][i], fontsize=label_fontsize)
        artist.axes[dim - 1, i].set_xlabel(plot_data["variable_names"][i], fontsize=label_fontsize)

    # Return figure
    artist.tight_layout()

    return artist
