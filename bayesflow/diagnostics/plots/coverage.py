from collections.abc import Sequence, Mapping

import matplotlib.pyplot as plt
import numpy as np

from bayesflow.utils import prepare_plot_data, add_titles_and_labels, prettify_subplots, compute_empirical_coverage


def coverage(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    difference: bool = False,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    figsize: Sequence[int] = None,
    label_fontsize: int = 16,
    legend_fontsize: int = 14,
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    legend_location: str = "lower right",
    color: str = "#132a70",
    num_col: int = None,
    num_row: int = None,
) -> plt.Figure:
    """
    Creates coverage plots showing empirical coverage of posterior credible intervals.

    The empirical coverage shows the coverage (proportion of true variable values that fall within the interval)
    of the central posterior credible intervals.
    A well-calibrated model would have coverage exactly match interval width (i.e. 95%
    credible interval contains the true value 95% of the time) as shown by the diagonal line.

    The coverage is accompanied by credible intervals for the coverage (gray ribbon).
    These are computed via the (conjugate) Beta-Binomial model for binomial proportions with a uniform prior.
    For more details on the Beta-Binomial model, see Chapter 2 of Bayesian Data Analysis (2013, 3rd ed.) by
    Gelman A., Carlin J., Stern H., Dunson D., Vehtari A., & Rubin D.

    Parameters
    ----------
    estimates : np.ndarray of shape (num_datasets, num_post_draws, num_params)
        The posterior draws obtained from num_datasets
    targets : np.ndarray of shape (num_datasets, num_params)
        The true parameter values used for generating num_datasets
    difference : bool, optional, default: True
        If True, plots the difference between empirical coverage and ideal coverage
        (coverage - width), making deviations from ideal calibration more visible.
        If False, plots the standard coverage plot.
    variable_keys : list or None, optional, default: None
        Select keys from the dictionaries provided in estimates and targets.
        By default, select all keys.
    variable_names : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    figsize : tuple or None, optional, default: None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize : int, optional, default: 16
        The font size of the y-label and x-label text
    legend_fontsize   : int, optional, default: 14
        The font size of the legend text
    title_fontsize : int, optional, default: 18
        The font size of the title text
    tick_fontsize : int, optional, default: 12
        The font size of the axis ticklabels
    legend_location : str, optional, default: 'upper right
        The location of the legend.
    color : str, optional, default: '#132a70'
        The color for the coverage line
    num_row : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    num_col : int, optional, default: None
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
    plot_data = prepare_plot_data(
        estimates=estimates,
        targets=targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
        num_col=num_col,
        num_row=num_row,
        figsize=figsize,
    )

    estimates = plot_data.pop("estimates")
    targets = plot_data.pop("targets")

    # Determine widths to compute coverage for
    num_draws = estimates.shape[1]
    widths = np.arange(0, num_draws + 2) / (num_draws + 1)

    # Compute empirical coverage with default parameters
    coverage_data = compute_empirical_coverage(
        estimates=estimates,
        targets=targets,
        widths=widths,
        prob=0.95,
        interval_type="central",
    )

    # Plot coverage for each parameter
    for i, ax in enumerate(plot_data["axes"].flat):
        if i >= plot_data["num_variables"]:
            break

        width_rep = coverage_data["width_represented"][:, i]
        coverage_est = coverage_data["coverage_estimates"][:, i]
        coverage_low = coverage_data["coverage_lower"][:, i]
        coverage_high = coverage_data["coverage_upper"][:, i]

        if difference:
            # Compute differences for coverage difference plot
            diff_est = coverage_est - width_rep
            diff_low = coverage_low - width_rep
            diff_high = coverage_high - width_rep

            # Plot confidence ribbon
            ax.fill_between(
                width_rep,
                diff_low,
                diff_high,
                color="grey",
                alpha=0.33,
                label="95% Credible Interval",
            )

            # Plot ideal coverage difference line (y = 0)
            ax.axhline(y=0, color="black", linestyle="dashed", label="Ideal Coverage")

            # Plot empirical coverage difference
            ax.plot(width_rep, diff_est, color=color, alpha=1.0, label="Coverage Difference")

        else:
            # Plot confidence ribbon
            ax.fill_between(
                width_rep,
                coverage_low,
                coverage_high,
                color="grey",
                alpha=0.33,
                label="95% Credible Interval",
            )

            # Plot ideal coverage line (y = x)
            ax.plot([0, 1], [0, 1], color="black", linestyle="dashed", label="Ideal Coverage")

            # Plot empirical coverage
            ax.plot(width_rep, coverage_est, color=color, alpha=1.0, label="Empirical Coverage")

        # Add legend to first subplot
        if i == 0:
            ax.legend(fontsize=legend_fontsize, loc=legend_location)

    prettify_subplots(plot_data["axes"], num_subplots=plot_data["num_variables"], tick_fontsize=tick_fontsize)

    # Add labels, titles, and set font sizes
    ylabel = "Empirical coverage difference" if difference else "Empirical coverage"
    add_titles_and_labels(
        axes=plot_data["axes"],
        num_row=plot_data["num_row"],
        num_col=plot_data["num_col"],
        title=plot_data["variable_names"],
        xlabel="Central interval width",
        ylabel=ylabel,
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )

    plot_data["fig"].tight_layout()
    return plot_data["fig"]
