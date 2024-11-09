import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import binom
from ..utils.plot_utils import check_posterior_prior_shapes, set_layout


def plot_sbc_histograms(
    post_samples,
    prior_samples,
    param_names=None,
    fig_size=None,
    num_bins=None,
    binomial_interval=0.99,
    label_fontsize=16,
    title_fontsize=18,
    tick_fontsize=12,
    hist_color="#a34f4f",
    n_row=None,
    n_col=None,
):
    """Creates and plots publication-ready histograms of rank statistics for simulation-based calibration
    (SBC) checks according to [1].

    Any deviation from uniformity indicates miscalibration and thus poor convergence
    of the networks or poor combination between generative model / networks.

    [1] Talts, S., Betancourt, M., Simpson, D., Vehtari, A., & Gelman, A. (2018).
    Validating Bayesian inference algorithms with simulation-based calibration.
    arXiv preprint arXiv:1804.06788.

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
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
    hist_color        : str, optional, default '#a34f4f'
        The color to use for the histogram body
    n_row             : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    n_col             : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of `post_samples` and `prior_samples`.
    """

    # Sanity check
    check_posterior_prior_shapes(post_samples, prior_samples)

    # Determine the ratio of simulations to prior draws
    n_sim, n_draws, n_params = post_samples.shape
    ratio = int(n_sim / n_draws)

    # Log a warning if N/B ratio recommended by Talts et al. (2018) < 20
    if ratio < 20:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.info(
            "The ratio of simulations / posterior draws should be > 20 "
            + f"for reliable variance reduction, but your ratio is {ratio}.\
                    Confidence intervals might be unreliable!"
        )

    # Set n_bins automatically, if nothing provided
    if num_bins is None:
        num_bins = int(ratio / 2)
        # Attempt a fix if a single bin is determined so plot still makes sense
        if num_bins == 1:
            num_bins = 5

    # Determine n params and param names if None given
    if param_names is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Determine number of rows and columns for subplots based on inputs
    if n_row is None or n_col is None:
        n_row, n_col = set_layout(n_total=n_params)

    # Initialize figure
    if fig_size is None:
        fig_size = (int(5 * n_col), int(5 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)
    axarr = np.atleast_1d(axarr)

    # Compute ranks (using broadcasting)
    ranks = np.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1)

    # Compute confidence interval and mean
    N = int(prior_samples.shape[0])
    # uniform distribution expected -> for all bins: equal probability
    # p = 1 / num_bins that a rank lands in that bin
    endpoints = binom.interval(binomial_interval, N, 1 / num_bins)
    mean = N / num_bins  # corresponds to binom.mean(N, 1 / num_bins)

    # Plot marginal histograms in a loop
    if n_row > 1:
        ax = axarr.flat
    else:
        ax = axarr
    for j in range(len(param_names)):
        ax[j].axhspan(endpoints[0], endpoints[1], facecolor="gray", alpha=0.3)
        ax[j].axhline(mean, color="gray", zorder=0, alpha=0.9)
        sns.histplot(ranks[:, j], kde=False, ax=ax[j], color=hist_color, bins=num_bins, alpha=0.95)
        ax[j].set_title(param_names[j], fontsize=title_fontsize)
        ax[j].spines["right"].set_visible(False)
        ax[j].spines["top"].set_visible(False)
        ax[j].get_yaxis().set_ticks([])
        ax[j].set_ylabel("")
        ax[j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax[j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Only add x-labels to the bottom row
    bottom_row = axarr if n_row == 1 else axarr[0] if n_col == 1 else axarr[n_row - 1, :]
    for _ax in bottom_row:
        _ax.set_xlabel("Rank statistic", fontsize=label_fontsize)

    # Remove unused axes entirely
    for _ax in axarr[n_params:]:
        _ax.remove()

    f.tight_layout()
    return f
