
import numpy as np
from scipy.stats import median_abs_deviation
from sklearn.metrics import r2_score
import seaborn as sns

from ..utils.plot_utils import preprocess, postprocess


def plot_recovery(
    post_samples,
    prior_samples,
    point_agg=np.median,
    uncertainty_agg=median_abs_deviation,
    param_names: list = None,
    fig_size: tuple = None,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    metric_fontsize: int = 16,
    tick_fontsize: int = 12,
    add_corr: bool = True,
    add_r2: bool = True,
    color: str | tuple = "#8f2727",
    n_col: int = None,
    n_row: int = None,
    xlabel: str = "Ground truth",
    ylabel: str = "Estimated",
    **kwargs,
):
    """Creates and plots publication-ready recovery plot with true vs. point estimate + uncertainty.
    The point estimate can be controlled with the ``point_agg`` argument, and the uncertainty estimate
    can be controlled with the ``uncertainty_agg`` argument.

    This plot yields similar information as the "posterior z-score", but allows for generic
    point and uncertainty estimates:

    https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html

    Important: Posterior aggregates play no special role in Bayesian inference and should only
    be used heuristically. For instance, in the case of multi-modal posteriors, common point
    estimates, such as mean, (geometric) median, or maximum a posteriori (MAP) mean nothing.

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws (true parameters) obtained for generating the n_data_sets
    point_agg         : callable, optional, default: ``np.median``
        The function to apply to the posterior draws to get a point estimate for each marginal.
        The default computes the marginal median for each marginal posterior as a robust
        point estimate.
    uncertainty_agg   : callable or None, optional, default: scipy.stats.median_abs_deviation
        The function to apply to the posterior draws to get an uncertainty estimate.
        If ``None`` provided, a simple scatter using only ``point_agg`` will be plotted.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 16
        The font size of the y-label text
    title_fontsize    : int, optional, default: 18
        The font size of the title text
    metric_fontsize   : int, optional, default: 16
        The font size of the goodness-of-fit metric (if provided)
    tick_fontsize     : int, optional, default: 12
        The font size of the axis tick labels
    add_corr          : bool, optional, default: True
        A flag for adding correlation between true and estimates to the plot
    add_r2            : bool, optional, default: True
        A flag for adding R^2 between true and estimates to the plot
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and error bars
    n_row             : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    n_col             : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.
    xlabel            : str, optional, default: 'Ground truth'
        The label on the x-axis of the plot
    ylabel            : str, optional, default: 'Estimated'
        The label on the y-axis of the plot
    **kwargs          : optional
        Additional keyword arguments passed to ax.errorbar or ax.scatter.
        Example: `rasterized=True` to reduce PDF file size with many dots

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation from the expected shapes of ``post_samples`` and ``prior_samples``.
    """

    # Preprocess
    f, axarr, axarr_it, n_row, n_col, n_params, param_names = preprocess(
        post_samples, prior_samples, fig_size=fig_size
    )

    # Compute point estimates and uncertainties
    est = point_agg(post_samples, axis=1)
    if uncertainty_agg is not None:
        u = uncertainty_agg(post_samples, axis=1)

    # Loop and plot
    for i, ax in enumerate(axarr_it):
        if i >= n_params:
            break

        # Add scatter and error bars
        if uncertainty_agg is not None:
            _ = ax.errorbar(prior_samples[:, i], est[:, i], yerr=u[:, i], fmt="o", alpha=0.5, color=color, **kwargs)
        else:
            _ = ax.scatter(prior_samples[:, i], est[:, i], alpha=0.5, color=color, **kwargs)

        # Make plots quadratic to avoid visual illusions
        lower = min(prior_samples[:, i].min(), est[:, i].min())
        upper = max(prior_samples[:, i].max(), est[:, i].max())
        eps = (upper - lower) * 0.1
        ax.set_xlim([lower - eps, upper + eps])
        ax.set_ylim([lower - eps, upper + eps])
        ax.plot(
            [ax.get_xlim()[0], ax.get_xlim()[1]],
            [ax.get_ylim()[0], ax.get_ylim()[1]],
            color="black",
            alpha=0.9,
            linestyle="dashed",
        )

        # Add optional metrics and title
        if add_r2:
            r2 = r2_score(prior_samples[:, i], est[:, i])
            ax.text(
                0.1,
                0.9,
                "$R^2$ = {:.3f}".format(r2),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        if add_corr:
            corr = np.corrcoef(prior_samples[:, i], est[:, i])[0, 1]
            ax.text(
                0.1,
                0.8,
                "$r$ = {:.3f}".format(corr),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )
        ax.set_title(param_names[i], fontsize=title_fontsize)

        # Prettify
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    postprocess(axarr, axarr_it, n_row, n_col, n_params, xlabel, ylabel, label_fontsize)

    f.tight_layout()
    return f
