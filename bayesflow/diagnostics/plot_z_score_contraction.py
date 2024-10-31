import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.plot_utils import check_posterior_prior_shapes


def plot_z_score_contraction(
        post_samples,
        prior_samples,
        param_names=None,
        fig_size=None,
        label_fontsize=16,
        title_fontsize=18,
        tick_fontsize=12,
        color="#8f2727",
        n_col=None,
        n_row=None,
):
    """
    Implements a graphical check for global model sensitivity by plotting the
    posterior z-score over the posterior contraction for each set of posterior
    samples in ``post_samples`` according to [1].

    - The definition of the posterior z-score is:

    post_z_score = (posterior_mean - true_parameters) / posterior_std

    And the score is adequate if it centers around zero and spreads roughly
    in the interval [-3, 3]

    - The definition of posterior contraction is:

    post_contraction = 1 - (posterior_variance / prior_variance)

    In other words, the posterior contraction is a proxy for the reduction in
    uncertainty gained by replacing the prior with the posterior.
    The ideal posterior contraction tends to 1.
    Contraction near zero indicates that the posterior variance is almost
    identical to the prior variance for the particular marginal parameter
    distribution.

    Note:
    Means and variances will be estimated via their sample-based estimators.

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

    # Sanity check for shape integrity
    check_posterior_prior_shapes(post_samples, prior_samples)

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

    # Determine number of params and param names if None given
    n_params = prior_samples.shape[-1]
    if param_names is None:
        param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Determine number of rows and columns for subplots based on inputs
    if n_row is None and n_col is None:
        n_row = int(np.ceil(n_params / 6))
        n_col = int(np.ceil(n_params / n_row))
    elif n_row is None and n_col is not None:
        n_row = int(np.ceil(n_params / n_col))
    elif n_row is not None and n_col is None:
        n_col = int(np.ceil(n_params / n_row))

    # Initialize figure
    if fig_size is None:
        fig_size = (int(4 * n_col), int(4 * n_row))
    f, axarr = plt.subplots(n_row, n_col, figsize=fig_size)

    # turn axarr into 1D list
    axarr = np.atleast_1d(axarr)
    if n_col > 1 or n_row > 1:
        axarr_it = axarr.flat
    else:
        axarr_it = axarr

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

    # Only add x-labels to the bottom row
    bottom_row = axarr if n_row == 1 else (
        axarr[0] if n_col == 1 else axarr[n_row - 1, :]
    )
    for _ax in bottom_row:
        _ax.set_xlabel("Posterior contraction", fontsize=label_fontsize)

    # Only add y-labels to right left-most row
    if n_row == 1:  # if there is only one row, the ax array is 1D
        axarr[0].set_ylabel("Posterior z-score", fontsize=label_fontsize)
    # If there is more than one row, the ax array is 2D
    else:
        for _ax in axarr[:, 0]:
            _ax.set_ylabel("Posterior z-score", fontsize=label_fontsize)

    # Remove unused axes entirely
    for _ax in axarr_it[n_params:]:
        _ax.remove()

    f.tight_layout()
    return f
