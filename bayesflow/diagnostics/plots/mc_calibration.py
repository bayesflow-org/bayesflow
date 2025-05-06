from collections.abc import Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt

from bayesflow.utils import (
    prepare_plot_data,
    add_titles_and_labels,
    add_metric,
    prettify_subplots,
)

from bayesflow.diagnostics.metrics import expected_calibration_error


def mc_calibration(
    pred_models: Mapping[str, np.ndarray] | np.ndarray,
    true_models: Mapping[str, np.ndarray] | np.ndarray,
    model_names: Sequence[str] = None,
    num_bins: int = 10,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    metric_fontsize: int = 14,
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
    pred_models       : np.ndarray of shape (num_data_sets, num_models)
        The predicted posterior model probabilities (PMPs) per data set.
    true_models       : np.ndarray of shape (num_data_sets, num_models)
        The one-hot-encoded true model indices per data set.
    model_names       : list or None, optional, default: None
        The model names for nice plot titles. Inferred if None.
    num_bins          : int, optional, default: 10
        The number of bins to use for the calibration curves (and marginal histograms).
    label_fontsize    : int, optional, default: 16
        The font size of the y-label and y-label texts
    metric_fontsize   : int, optional, default: 14
        The font size of the metric (e.g., ECE)
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

    # Gather plot data and metadata into a dictionary
    plot_data = prepare_plot_data(
        estimates=pred_models,
        targets=true_models,
        variable_names=model_names,
        num_col=num_col,
        num_row=num_row,
        figsize=figsize,
        default_name="M",
    )

    # compute ece and probs
    ece = expected_calibration_error(
        estimates=pred_models,
        targets=true_models,
        model_names=plot_data["variable_names"],
        num_bins=num_bins,
        return_probs=True,
    )

    for j, ax in enumerate(plot_data["axes"].flat):
        # Plot calibration curve
        ax.plot(ece["probs_pred"][j], ece["probs_true"][j], "o-", color=color)

        # Plot PMP distribution over bins
        uniform_bins = np.linspace(0.0, 1.0, num_bins + 1)
        norm_weights = np.ones_like(plot_data["estimates"]) / len(plot_data["estimates"])
        ax.hist(plot_data["estimates"][:, j], bins=uniform_bins, weights=norm_weights[:, j], color="grey", alpha=0.3)

        # Plot AB line
        ax.plot((0, 1), (0, 1), "--", color="black", alpha=0.9)

        # Tweak plot
        ax.set_xlim([0 - epsilon, 1 + epsilon])
        ax.set_ylim([0 - epsilon, 1 + epsilon])
        ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

        # Add ECE label
        add_metric(
            ax,
            metric_text=r"$\widehat{{\mathrm{{ECE}}}}$",
            metric_value=ece["values"][j],
            metric_fontsize=metric_fontsize,
        )

    prettify_subplots(axes=plot_data["axes"], num_subplots=plot_data["num_variables"], tick_fontsize=tick_fontsize)

    # Add labels, titles, and set font sizes
    add_titles_and_labels(
        axes=plot_data["axes"],
        num_row=plot_data["num_row"],
        num_col=plot_data["num_col"],
        title=plot_data["variable_names"],
        xlabel="Predicted Probability",
        ylabel="True Probability",
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )

    plot_data["fig"].tight_layout()
    return plot_data["fig"]
