from collections.abc import Sequence, Mapping

import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from ...utils.plot_utils import make_figure
from ...utils.classification import confusion_matrix


def mc_confusion_matrix(
    pred_models: Mapping[str, np.ndarray] | np.ndarray,
    true_models: Mapping[str, np.ndarray] | np.ndarray,
    model_names: Sequence[str] = None,
    fig_size: tuple = (5, 5),
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    value_fontsize: int = 10,
    tick_fontsize: int = 12,
    xtick_rotation: int = None,
    ytick_rotation: int = None,
    normalize: str = None,
    cmap: matplotlib.colors.Colormap | str = None,
    title: bool = True,
) -> plt.Figure:
    """Plots a confusion matrix for validating a neural network trained for Bayesian model comparison.

    Parameters
    ----------
    pred_models    : np.ndarray of shape (num_data_sets, num_models)
        The predicted posterior model probabilities (PMPs) per data set.
    true_models    : np.ndarray of shape (num_data_sets, num_models)
        The one-hot-encoded true model indices per data set.
    model_names    : list or None, optional, default: None
        The model names for nice plot titles. Inferred if None.
    fig_size       : tuple or None, optional, default: (5, 5)
        The figure size passed to the ``matplotlib`` constructor. Inferred if ``None``
    label_fontsize    : int, optional, default: 16
        The font size of the y-label and y-label texts
    title_fontsize : int, optional, default: 18
        The font size of the title text.
    value_fontsize  : int, optional, default: 10
        The font size of the text annotations and the colorbar tick labels.
    tick_fontsize  : int, optional, default: 12
        The font size of the axis label and model name texts.
    xtick_rotation: int, optional, default: None
        Rotation of x-axis tick labels (helps with long model names).
    ytick_rotation: int, optional, default: None
        Rotation of y-axis tick labels (helps with long model names).
    normalize : {'true', 'pred', 'all'}, default=None
        Passed to confusion matrix. Normalizes confusion matrix over the true (rows),
        predicted (columns) conditions or all the population. If None, confusion matrix
        will not be normalized.
    cmap           : matplotlib.colors.Colormap or str, optional, default: None
        Colormap to be used for the cells. If a str, it should be the name of a registered colormap,
        e.g., 'viridis'. Default colormap matches the BayesFlow defaults by ranging from white to red.
    title          : bool, optional, default True
        A flag for adding 'Confusion Matrix' above the matrix.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving
    """

    if model_names is None:
        num_models = true_models.shape[-1]
        model_names = [rf"$M_{{{m}}}$" for m in range(1, num_models + 1)]

    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("", ["white", "#132a70"])

    # Flatten input
    true_models = np.argmax(true_models, axis=1)
    pred_models = np.argmax(pred_models, axis=1)

    # Compute confusion matrix
    cm = confusion_matrix(true_models, pred_models, normalize=normalize)

    # Initialize figure
    fig, ax = make_figure(1, 1, figsize=fig_size)
    ax = ax[0]
    im = ax.imshow(cm, interpolation="nearest", cmap=cmap)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75)

    cbar.ax.tick_params(labelsize=value_fontsize)

    ax.set_xticks(range(cm.shape[0]))
    ax.set_xticklabels(model_names, fontsize=tick_fontsize)
    if xtick_rotation:
        plt.xticks(rotation=xtick_rotation, ha="right")

    ax.set_yticks(range(cm.shape[1]))
    ax.set_yticklabels(model_names, fontsize=tick_fontsize)
    if ytick_rotation:
        plt.yticks(rotation=ytick_rotation)

    ax.set_xlabel("Predicted model", fontsize=label_fontsize)
    ax.set_ylabel("True model", fontsize=label_fontsize)

    # Loop over data dimensions and create text annotations
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                fontsize=value_fontsize,
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    if title:
        ax.set_title("Confusion Matrix", fontsize=title_fontsize)
    return fig
