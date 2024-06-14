
from ..utils.plotutils import preprocess, postprocess
from ..utils.computils import expected_calibration_error
from keras import ops


def plot_calibration_curves(
        true_models,
        pred_models,
        model_names: list = None,
        num_bins: int = 10,
        label_fontsize: int = 16,
        legend_fontsize: int = 14,
        title_fontsize: int = 18,
        tick_fontsize: int = 12,
        epsilon: float = 0.02,
        fig_size: tuple = None,
        color: str | tuple = "#8f2727",
        x_label: str = "Predicted probability",
        y_label: str = "True probability",
        n_row: int = None,
        n_col: int = None,
):
    """Plots the calibration curves, the ECEs and the marginal histograms of predicted posterior model probabilities
    for a model comparison problem. The marginal histograms inform about the fraction of predictions in each bin.
    Depends on the ``expected_calibration_error`` function for computing the ECE.

    Parameters
    ----------
    true_models       : np.ndarray of shape (num_data_sets, num_models)
        The one-hot-encoded true model indices per data set.
    pred_models       : np.ndarray of shape (num_data_sets, num_models)
        The predicted posterior model probabilities (PMPs) per data set.
    model_names       : list or None, optional, default: None
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
    fig_size          : tuple or None, optional, default: None
        The figure size passed to the ``matplotlib`` constructor. Inferred if ``None``
    color             : str, optional, default: '#8f2727'
        The color of the calibration curves
    x_label            : str, optional, default: Predicted probability
        The x-axis label
    y_label            : str, optional, default: True probability
        The y-axis label
    n_row             : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    n_col             : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.

    Returns
    -------
    fig : plt.Figure - the figure instance for optional saving
    """

    f, axarr, ax, n_row, n_col, num_models, model_names = preprocess(true_models, pred_models, fig_size=fig_size)

    # Compute calibration
    cal_errs, probs_true, probs_pred = expected_calibration_error(true_models, pred_models, num_bins)

    # Plot marginal calibration curves in a loop
    for j in range(num_models):
        # Plot calibration curve
        ax[j].plot(probs_pred[j], probs_true[j], "o-", color=color)

        # Plot PMP distribution over bins
        uniform_bins = ops.linspace(0.0, 1.0, num_bins + 1)
        norm_weights = ops.ones_like(pred_models) / len(pred_models)
        ax[j].hist(pred_models[:, j], bins=uniform_bins, weights=norm_weights[:, j], color="grey", alpha=0.3)

        # Plot AB line
        ax[j].plot((0, 1), (0, 1), "--", color="black", alpha=0.9)

        # Tweak plot
        ax[j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax[j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)
        ax[j].set_title(model_names[j], fontsize=title_fontsize)
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

    # Post-processing
    postprocess(axarr, ax, n_row, n_col, num_models, x_label, y_label, label_fontsize)

    f.tight_layout()
    return f
