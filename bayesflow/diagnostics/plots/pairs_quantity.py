from collections.abc import Sequence, Mapping

import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from bayesflow.utils.dict_utils import make_variable_array


def pairs_quantity(
    values: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    height: float = 2.5,
    cmap: str | matplotlib.colors.Colormap = "viridis",
    alpha: float = 0.9,
    label: str = "",
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    colorbar_label_fontsize: int = 14,
    colorbar_tick_fontsize: int = 12,
    colorbar_width: float = 1.8,
    colorbar_height: float = 0.06,
    colorbar_offset: float = 0.06,
    vmin: float = None,
    vmax: float = None,
    **kwargs,
) -> sns.PairGrid:
    """
    A pair plot function to plot quantities against their generating
    parameter values.

    The value is indicated by a colormap. The marginal distribution for
    each parameter is plotted on the diagonal. Each column displays the
    values of corresponding to the parameter in the column.

    Parameters
    ----------
    values      : dict[str, np.ndarray],
        The value of the quantity to plot.
    targets     : dict[str, np.ndarray],
        The parameter values plotted on the axis.
    variable_keys       : list or None, optional, default: None
       Select keys from the dictionary provided in samples.
       By default, select all keys.
    variable_names    : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    height      : float, optional, default: 2.5
        The height of the pair plot
    cmap       : str or Colormap, default: "viridis"
        The colormap for the plot.
    alpha       : float in [0, 1], optional, default: 0.9
        The opacity of the plot
    label       : str, optional, default: ""
        Label for the dataset to plot
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
    **kwargs    : dict, optional
        Additional keyword arguments passed to the sns.PairGrid constructor
    """
    targets = make_variable_array(
        targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
    )

    values = make_variable_array(
        values,
        variable_keys=variable_keys,
        variable_names=variable_names,
    )
    variable_names = values.variable_names

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
            if i == j:
                ax = g.axes[i, j].twinx()
                ax.scatter(
                    targets[:, i], values[:, i], c=values[:, i], cmap=cmap, s=4, vmin=vmin, vmax=vmax, alpha=alpha
                )
                ax.spines["left"].set_visible(False)
                ax.spines["top"].set_visible(False)
                ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
                ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)
                ax.set_ylim(vmin, vmax)

                if i > 0:
                    g.axes[i, j].get_yaxis().set_visible(False)
                    g.axes[i, j].spines["left"].set_visible(False)
            else:
                g.axes[i, j].grid(alpha=0.5)
                g.axes[i, j].set_axisbelow(True)
                g.axes[i, j].scatter(
                    targets[:, j],
                    targets[:, i],
                    c=values[:, j],
                    cmap=cmap,
                    s=4,
                    vmin=vmin,
                    vmax=vmax,
                    alpha=alpha,
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

    return g.figure
