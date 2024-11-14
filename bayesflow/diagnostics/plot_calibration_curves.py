import matplotlib.pyplot as plt
import numpy as np

from typing import Sequence
from ..utils.comp_utils import expected_calibration_error
from ..utils.plot_utils import preprocess, add_labels


def plot_calibration_curves(
    post_model_samples: dict[str, np.ndarray] | np.ndarray,
    true_model_samples: dict[str, np.ndarray] | np.ndarray,
    names: Sequence[str] = None,
    num_bins: int = 10,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    legend_fontsize: int = 14,
    tick_fontsize: int = 12,
    epsilon: float = 0.02,
    figsize: Sequence[int] = None,
    color: str = "#132a70",
    num_col: int = None,
    num_row: int = None,
) -> plt.Figure:
    """Plots the calibration curves, the ECEs and the marginal histograms of predicted posterior model probabilities
    for a model comparison problem. The marginal histograms inform about the fraction of predictions in each bin.
    Depends on the ``expected_calibration_error`` function for computing the ECE.

    Parameters
    ----------
    true_model_samples       : np.ndarray of shape (num_data_sets, num_models)
        The one-hot-encoded true model indices per data set.
    post_model_samples      : np.ndarray of shape (num_data_sets, num_models)
        The predicted posterior model probabilities (PMPs) per data set.
    names       : list or None, optional, default: None
        The model names for nice plot titles. Inferred if None.
    num_bins          : int, optional, default: 10
        The number of bins to use for the calibration curves (and marginal histograms).
    label_fontsize    : int, optional, default: 16
        The font size of the y-label and y-label texts
    legend_fontsize   : int, optional, default: 14
        The font size of the legend text (ECE value)
    title_fontsize    : int, optional, default: 18
        The font size of the title text. Only relevant if `stacked=False`
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    epsilon           : float, optional, default: 0.02
        A small amount to pad the [0, 1]-bounded axes from both side.
    figsize          : tuple or None, optional, default: None
        The figure size passed to the ``matplotlib`` constructor. Inferred if ``None``
    color             : str, optional, default: '#8f2727'
        The color of the calibration curves
    num_row             : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    num_col             : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving
    """

    plot_data = preprocess(post_model_samples, true_model_samples, names, num_col, num_row, figsize, context="M")

    # Plot marginal calibration curves in a loop
    if plot_data['num_row'] > 1:
        ax = plot_data['axes'].flat
    else:
        ax = plot_data['axes']

    # Compute calibration
    cal_errs, probs_true, probs_pred = expected_calibration_error(
        plot_data['prior_samples'], plot_data['post_samples'], num_bins)

    for j in range(plot_data['num_variables']):
        # Plot calibration curve
        ax[j].plot(probs_pred[j], probs_true[j], "o-", color=color)

        # Plot PMP distribution over bins
        uniform_bins = np.linspace(0.0, 1.0, num_bins + 1)
        norm_weights = np.ones_like(plot_data['post_samples']) / len(plot_data['post_samples'])
        ax[j].hist(plot_data['post_samples'][:, j], bins=uniform_bins, weights=norm_weights[:, j], color="grey", alpha=0.3)

        # Plot AB line
        ax[j].plot((0, 1), (0, 1), "--", color="black", alpha=0.9)

        # Tweak plot
        ax[j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax[j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)
        ax[j].set_title(plot_data['names'][j], fontsize=title_fontsize)
        ax[j].spines["right"].set_visible(False)
        ax[j].spines["top"].set_visible(False)
        ax[j].set_xlim([0 - epsilon, 1 + epsilon])
        ax[j].set_ylim([0 - epsilon, 1 + epsilon])
        ax[j].set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax[j].set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax[j].grid(alpha=0.5)

        # Add ECE label
        ax[j].text(
            0.1,
            0.9,
            r"$\widehat{{\mathrm{{ECE}}}}$ = {0:.3f}".format(cal_errs[j]),
            horizontalalignment="left",
            verticalalignment="center",
            transform=ax[j].transAxes,
            size=legend_fontsize,
        )

    # Only add x-labels to the bottom row
    add_labels(
        axes=plot_data['axes'],
        num_row=plot_data['num_row'],
        num_col=plot_data['num_col'],
        xlabel="Predicted Probability",
        ylabel="True Probability",
        label_fontsize=label_fontsize,
    )

    plot_data['fig'].tight_layout()
    return plot_data['fig']
