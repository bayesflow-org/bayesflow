import logging
import seaborn as sns

from scipy.stats import binom
from keras import ops
from keras import backend as K
from ..utils.plotutils import preprocess, remove_unused_axes


def plot_sbc_histograms(
    post_samples,
    prior_samples,
    param_names: list = None,
    fig_size: tuple = None,
    num_bins: int = None,
    binomial_interval: float = 0.99,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    hist_color: str | tuple = "#a34f4f",
    n_row: int = None,
    n_col: int = None,
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

    f, axarr, ax, n_row, n_col, n_params, param_names = preprocess(post_samples, prior_samples, fig_size=fig_size)

    # Determine the ratio of simulations to prior draws
    n_sim, n_draws, _ = post_samples.shape
    ratio = int(n_sim / n_draws)

    # Log a warning if N/B ratio recommended by Talts et al. (2018) < 20
    if ratio < 20:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.info(
            f"The ratio of simulations / posterior draws should be > 20 "
            + f"for reliable variance reduction, but your ratio is {ratio}.\
                    Confidence intervals might be unreliable!"
        )

    # Set n_bins automatically, if nothing provided
    if num_bins is None:
        num_bins = int(ratio / 2)
        # Attempt a fix if a single bin is determined so plot still makes sense
        if num_bins == 1:
            num_bins = 5
    
    # Compute ranks (using broadcasting)
    post_samples = K.constant(post_samples)
    prior_samples = K.constant(prior_samples)

    # Adding an extra dimension to prior_samples using K.expand_dims
    prior_samples_expanded = K.expand_dims(prior_samples, axis=1)

    # Performing element-wise comparison
    comparison = K.less(post_samples, prior_samples_expanded)

    # Summing along the specified axis (axis=1)
    ranks = K.sum(K.cast(comparison, dtype='float32'), axis=1)
    # ranks = ops.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1)

    # Compute confidence interval and mean
    N = int(prior_samples.shape[0])
    # uniform distribution expected -> for all bins: equal probability
    # p = 1 / num_bins that a rank lands in that bin
    endpoints = binom.interval(binomial_interval, N, 1 / num_bins)
    mean = N / num_bins  # corresponds to binom.mean(N, 1 / num_bins)

    # Plot marginal histograms in a loop
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
    remove_unused_axes(axarr, n_params)

    f.tight_layout()
    return f
