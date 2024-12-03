import numpy as np
import matplotlib.pyplot as plt

from typing import Sequence

from bayesflow.utils import preprocess, add_titles_and_labels, prettify_subplots


def z_score_contraction(
    post_samples: dict[str, np.ndarray] | np.ndarray,
    prior_samples: dict[str, np.ndarray] | np.ndarray,
    filter_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    figsize: Sequence[int] = None,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    color: str = "#132a70",
    num_col: int = None,
    num_row: int = None,
) -> plt.Figure:
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
    variable_names    : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    figsize           : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 16
        The font size of the y-label text
    title_fontsize    : int, optional, default: 18
        The font size of the title text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and error bars
    num_row           : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    num_col           : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation from the expected shapes of ``post_samples`` and ``prior_samples``.
    """

    # Preprocessing
    plot_data = preprocess(post_samples, prior_samples, filter_keys, variable_names, num_col, num_row, figsize)
    plot_data["post_samples"] = plot_data.pop("post_variables")
    plot_data["prior_samples"] = plot_data.pop("prior_variables")

    # Estimate posterior means and stds
    post_means = plot_data["post_samples"].mean(axis=1)
    post_stds = plot_data["post_samples"].std(axis=1, ddof=1)
    post_vars = plot_data["post_samples"].var(axis=1, ddof=1)

    # Estimate prior variance
    prior_vars = plot_data["prior_samples"].var(axis=0, keepdims=True, ddof=1)

    # Compute contraction and z-score
    contraction = 1 - (post_vars / prior_vars)
    z_score = (post_means - plot_data["prior_samples"]) / post_stds

    # Loop and plot
    for i, ax in enumerate(plot_data["axes"].flat):
        if i >= plot_data["num_variables"]:
            break

        ax.scatter(contraction[:, i], z_score[:, i], color=color, alpha=0.5)
        ax.set_xlim([-0.05, 1.05])

    prettify_subplots(plot_data["axes"], tick_fontsize)

    # Add labels, titles, and set font sizes
    add_titles_and_labels(
        axes=plot_data["axes"],
        num_row=plot_data["num_row"],
        num_col=plot_data["num_col"],
        title=plot_data["variable_names"],
        xlabel="Posterior contraction",
        ylabel="Posterior z-score",
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )

    plot_data["fig"].tight_layout()
    return plot_data["fig"]
