from collections.abc import Sequence, Mapping

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from bayesflow.utils import logging
from bayesflow.utils.dict_utils import dicts_to_arrays
from bayesflow.utils.plot_utils import create_legends


def pairs_samples(
    samples: Mapping[str, np.ndarray] | np.ndarray = None,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    height: float = 2.5,
    color: str | tuple = "#132a70",
    alpha: float = 0.9,
    label: str = "Posterior",
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    show_single_legend: bool = False,
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
        The primary color of the plot
    alpha       : float in [0, 1], optional, default: 0.9
        The opacity of the plot
    label       : str, optional, default: "Posterior"
        Label for the dataset to plot
    label_fontsize    : int, optional, default: 14
        The font size of the x and y-label texts (parameter names)
    tick_fontsize     : int, optional, default: 12
        The font size of the axis tick labels
    show_single_legend : bool, optional, default: False
        Optional toggle for the user to choose whether a single dataset
        should also display legend
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
        label=label,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        show_single_legend=show_single_legend,
        **kwargs,
    )

    return g


def _pairs_samples(
    plot_data: dict,
    height: float = 2.5,
    color: str | tuple = "#132a70",
    color2: str | tuple = "gray",
    alpha: float = 0.9,
    label: str = "Posterior",
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    legend_fontsize: int = 14,
    show_single_legend: bool = False,
    **kwargs,
) -> sns.PairGrid:
    """
    Internal version of pairs_samples creating the seaborn PairPlot
    for both a single dataset and multiple datasets.

    Parameters
    ----------
    plot_data   : output of bayesflow.utils.dict_utils.dicts_to_arrays
        Formatted data to plot from the sample dataset
    color2      : str, optional, default: 'gray'
        Secondary color for the pair plots.
        This is the color used for the prior draws.

    Other arguments are documented in pairs_samples
    """

    estimates_shape = plot_data["estimates"].shape
    if len(estimates_shape) != 2:
        raise ValueError(
            f"Samples for a single distribution should be a matrix, but "
            f"your samples array has a shape of {estimates_shape}."
        )

    variable_names = plot_data["estimates"].variable_names

    # Convert samples to pd.DataFrame
    if plot_data["priors"] is not None:
        # differentiate posterior from prior draws
        # row bind posterior and prior draws
        samples = np.vstack((plot_data["priors"], plot_data["estimates"]))
        data_to_plot = pd.DataFrame(samples, columns=variable_names)

        # ensure that the source of the samples is stored
        source_prior = np.repeat("Prior", plot_data["priors"].shape[0])
        source_post = np.repeat("Posterior", plot_data["estimates"].shape[0])
        data_to_plot["_source"] = np.concatenate((source_prior, source_post))
        data_to_plot["_source"] = pd.Categorical(data_to_plot["_source"], categories=["Prior", "Posterior"])

        # initialize plot
        g = sns.PairGrid(
            data_to_plot,
            height=height,
            hue="_source",
            palette=[color2, color],
            **kwargs,
        )

        # ensures that color doesn't overwrite palette
        color = None

    else:
        # plot just the one set of distributions
        data_to_plot = pd.DataFrame(plot_data["estimates"], columns=variable_names)

        # initialize plot
        g = sns.PairGrid(data_to_plot, height=height, **kwargs)

    # add histograms + KDEs to the diagonal
    g.map_diag(
        histplot_twinx,
        fill=True,
        kde=True,
        color=color,
        alpha=alpha,
        stat="density",
        common_norm=False,
    )

    # add scatter plots to the upper diagonal
    g.map_upper(sns.scatterplot, alpha=0.6, s=40, edgecolor="k", color=color, lw=0)

    # add KDEs to the lower diagonal
    try:
        g.map_lower(sns.kdeplot, fill=True, color=color, alpha=alpha)
    except Exception as e:
        logging.exception("KDE failed due to the following exception:\n" + repr(e) + "\nSubstituting scatter plot.")
        g.map_lower(sns.scatterplot, alpha=0.6, s=40, edgecolor="k", color=color, lw=0)

    # Generate grids
    dim = g.axes.shape[0]
    for i in range(dim):
        for j in range(dim):
            g.axes[i, j].grid(alpha=0.5)
            g.axes[i, j].set_axisbelow(True)

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

    # need to add legend here such that colors are recognized
    # if plot_data["priors"] is not None:
    #     g.add_legend(fontsize=legend_fontsize, loc="center right")
    #     g._legend.set_title(None)

    create_legends(
        g,
        plot_data,
        color=color,
        color2=color2,
        legend_fontsize=legend_fontsize,
        label=label,
        show_single_legend=show_single_legend,
    )

    # Return figure
    g.tight_layout()

    return g


def histplot_twinx(x, **kwargs):
    """
    # create a histogram plot on a twin y-axis
    # this ensures that the y scaling of the diagonal plots
    # in independent of the y scaling of the off-diagonal plots

    Parameters
    ----------
    x : np.ndarray
        Data to be plotted.
    """
    # create a histogram on the twin axis
    sns.histplot(x, legend=False, **kwargs)

    # make the twin axis invisible
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)

    return None
