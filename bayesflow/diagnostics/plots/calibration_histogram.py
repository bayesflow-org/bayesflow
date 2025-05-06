from collections.abc import Sequence, Mapping

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from scipy.stats import binom

from bayesflow.utils import logging
from bayesflow.utils import prepare_plot_data, add_titles_and_labels, prettify_subplots


def calibration_histogram(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    figsize: Sequence[float] = None,
    num_bins: int = 10,
    binomial_interval: float = 0.99,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    color: str = "#132a70",
    num_col: int = None,
    num_row: int = None,
) -> plt.Figure:
    """Creates and plots publication-ready histograms of rank statistics for simulation-based calibration
    (SBC) checks according to [1].

    Any deviation from uniformity indicates miscalibration and thus poor convergence
    of the networks or poor combination between generative model / networks.

    [1] Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
    Validating Bayesian inference algorithms with simulation-based calibration.
    arXiv preprint arXiv:1804.06788.

    Parameters
    ----------
    estimates      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    targets     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets
    variable_keys       : list or None, optional, default: None
       Select keys from the dictionaries provided in estimates and targets.
       By default, select all keys.
    variable_names    : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    figsize          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None
    num_bins          : int, optional, default: 10
        The number of bins to use for each marginal histogram
    binomial_interval : float in (0, 1), optional, default: 0.99
        The width of the confidence interval for the binomial distribution
    label_fontsize    : int, optional, default: 16
        The font size of the y-label text
    title_fontsize    : int, optional, default: 18
        The font size of the title text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    color        : str, optional, default '#a34f4f'
        The color to use for the histogram body
    num_row             : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    num_col             : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of `estimates` and `targets`.
    """

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

    # Determine the ratio of simulations to prior draw
    # num_params = plot_data['num_variables']
    num_sims = estimates.shape[0]
    num_draws = estimates.shape[1]

    ratio = int(num_sims / num_draws)

    # Log a warning if N/B ratio recommended by Talts et al. (2018) < 20
    if ratio < 20:
        logging.warning(
            "The ratio of simulations / posterior draws should be > 20 "
            f"for reliable variance reduction, but your ratio is {ratio}. "
            "Confidence intervals might be unreliable!"
        )

    # Set num_bins automatically, if nothing provided
    if num_bins is None:
        num_bins = int(ratio / 2)
        # Attempt a fix if a single bin is determined so plot still makes sense
        if num_bins == 1:
            num_bins = 4

    # Compute ranks (using broadcasting)
    ranks = np.sum(estimates < targets[:, np.newaxis, :], axis=1)

    # Compute confidence interval and mean
    num_trials = int(targets.shape[0])
    # uniform distribution expected -> for all bins: equal probability
    # p = 1 / num_bins that a rank lands in that bin
    endpoints = binom.interval(binomial_interval, num_trials, 1 / num_bins)
    mean = num_trials / num_bins  # corresponds to binom.mean(N, 1 / num_bins)

    for j, ax in enumerate(plot_data["axes"].flat):
        ax.axhspan(endpoints[0], endpoints[1], facecolor="gray", alpha=0.3)
        ax.axhline(mean, color="gray", zorder=0, alpha=0.9)
        sns.histplot(ranks[:, j], kde=False, ax=ax, color=color, bins=num_bins, alpha=0.95)
        ax.get_yaxis().set_ticks([])
    prettify_subplots(plot_data["axes"], tick_fontsize)

    add_titles_and_labels(
        axes=plot_data["axes"],
        num_row=plot_data["num_row"],
        num_col=plot_data["num_col"],
        title=plot_data["variable_names"],
        xlabel="Rank statistic",
        ylabel="",
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )
    plot_data["fig"].tight_layout()

    return plot_data["fig"]
