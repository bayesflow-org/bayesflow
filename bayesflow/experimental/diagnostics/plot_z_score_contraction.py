import seaborn as sns
from ..utils.plotutils import preprocess, postprocess


def plot_z_score_contraction(
    post_samples,
    prior_samples,
    param_names: list = None,
    fig_size: tuple = None,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    color: str | tuple = "#8f2727",
    x_label: str = "Posterior contraction",
    y_label: str = "Posterior z-score",
    n_col: int = None,
    n_row: int = None,
):
    """Implements a graphical check for global model sensitivity by plotting the posterior
    z-score over the posterior contraction for each set of posterior samples in ``post_samples``
    according to [1].

    - The definition of the posterior z-score is:

    post_z_score = (posterior_mean - true_parameters) / posterior_std

    And the score is adequate if it centers around zero and spreads roughly in the interval [-3, 3]

    - The definition of posterior contraction is:

    post_contraction = 1 - (posterior_variance / prior_variance)

    In other words, the posterior contraction is a proxy for the reduction in uncertainty gained by
    replacing the prior with the posterior. The ideal posterior contraction tends to 1.
    Contraction near zero indicates that the posterior variance is almost identical to
    the prior variance for the particular marginal parameter distribution.

    Note: Means and variances will be estimated via their sample-based estimators.

    [1] Schad, D. J., Betancourt, M., & Vasishth, S. (2021).
    Toward a principled Bayesian workflow in cognitive science.
    Psychological methods, 26(1), 103.

    Paper also available at https://arxiv.org/abs/1904.12765

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws (true parameters) obtained for generating the n_data_sets
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    fig_size          : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 16
        The font size of the y-label text
    title_fontsize    : int, optional, default: 18
        The font size of the title text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and error bars
    x_label           : str, optional, default: Posterior contraction
        The label for the x-axis
    y_label           : str, optional, default: Posterior z-score
        The label for the y-axis
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
        If there is a deviation from the expected shapes of ``post_samples`` and ``prior_samples``.
    """

    f, axarr, axarr_it, n_row, n_col, n_params, param_names = preprocess(post_samples, prior_samples, fig_size=fig_size)

    # Estimate posterior means and stds
    post_means = post_samples.mean(axis=1)
    post_stds = post_samples.std(axis=1, ddof=1)
    post_vars = post_samples.var(axis=1, ddof=1)

    # Estimate prior variance
    prior_vars = prior_samples.var(axis=0, keepdims=True, ddof=1)

    # Compute contraction
    post_cont = 1 - (post_vars / prior_vars)

    # Compute posterior z score
    z_score = (post_means - prior_samples) / post_stds

    # Loop and plot
    for i, ax in enumerate(axarr_it):
        if i >= n_params:
            break

        ax.scatter(post_cont[:, i], z_score[:, i], color=color, alpha=0.5)
        ax.set_title(param_names[i], fontsize=title_fontsize)
        sns.despine(ax=ax)
        ax.grid(alpha=0.5)
        ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)
        ax.set_xlim([-0.05, 1.05])

    postprocess(axarr, axarr_it, n_row, n_col, n_params, x_label, y_label, label_fontsize)

    f.tight_layout()
    return f
