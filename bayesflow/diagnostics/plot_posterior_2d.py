import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.lines import Line2D
from .plot_samples_2d import plot_samples_2d


def plot_posterior_2d(
    post_samples: dict[str, np.ndarray] | np.ndarray,
    prior_samples: dict[str, np.ndarray] | np.ndarray,
    prior=None,
    param_names: list = None,
    height: int = 3,
    label_fontsize: int = 14,
    legend_fontsize: int = 16,
    tick_fontsize: int = 12,
    post_color: str | tuple = "#132a70",
    prior_color: str | tuple = "gray",
    post_alpha: float = 0.9,
    prior_alpha: float = 0.7,
    **kwargs,
) -> plt.Figure:
    """Generates a bivariate pairplot given posterior draws and optional prior or prior draws.

    posterior_draws   : np.ndarray of shape (n_post_draws, n_params)
        The posterior draws obtained for a SINGLE observed data set.
    prior             : bayesflow.forward_inference.Prior instance or None, optional, default: None
        The optional prior object having an input-output signature as given by ayesflow.forward_inference.Prior
    prior_draws       : np.ndarray of shape (n_prior_draws, n_params) or None, optonal (default: None)
        The optional prior draws obtained from the prior. If both prior and prior_draws are provided, prior_draws
        will be used.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    height            : float, optional, default: 3
        The height of the pairplot
    label_fontsize    : int, optional, default: 14
        The font size of the x and y-label texts (parameter names)
    legend_fontsize   : int, optional, default: 16
        The font size of the legend text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    post_color        : str, optional, default: '#8f2727'
        The color for the posterior histograms and KDEs
    priors_color      : str, optional, default: gray
        The color for the optional prior histograms and KDEs
    post_alpha        : float in [0, 1], optonal, default: 0.9
        The opacity of the posterior plots
    prior_alpha       : float in [0, 1], optonal, default: 0.7
        The opacity of the prior plots

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    AssertionError
        If the shape of posterior_draws is not 2-dimensional.
    """

    # Ensure correct shape
    assert (len(post_samples.shape)) == 2, "Shape of `posterior_samples` for a single data set should be 2 dimensional!"

    # Plot posterior first
    g = plot_samples_2d(post_samples, context="\\theta", param_names=param_names, render=False, height=height, **kwargs)

    # Obtain n_draws and n_params
    n_draws, n_params = post_samples.shape

    # If prior object is given and no draws, obtain draws
    if prior is not None and prior_samples is None:
        draws = prior(n_draws)
        if isinstance(draws, dict):
            prior_draws = draws["prior_draws"]
        else:
            prior_draws = draws

    # Attempt to determine parameter names
    if param_names is None:
        if hasattr(prior, "param_names"):
            if prior.param_names is not None:
                param_names = prior.param_names
            else:
                param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]
        else:
            param_names = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]

    # Add prior, if given
    if prior_draws is not None:
        prior_draws_df = pd.DataFrame(prior_draws, columns=param_names)
        g.data = prior_draws_df
        g.map_diag(sns.histplot, fill=True, color=prior_color, alpha=prior_alpha, kde=True, zorder=-1)
        g.map_lower(sns.kdeplot, fill=True, color=prior_color, alpha=prior_alpha, zorder=-1)

    # Add legend, if prior also given
    if prior_draws is not None or prior is not None:
        handles = [
            Line2D(xdata=[], ydata=[], color=post_color, lw=3, alpha=post_alpha),
            Line2D(xdata=[], ydata=[], color=prior_color, lw=3, alpha=prior_alpha),
        ]
        g.legend(handles, ["Posterior", "Prior"], fontsize=legend_fontsize, loc="center right")

    n_row, n_col = g.axes.shape

    for i in range(n_row):
        # Remove upper axis
        for j in range(i + 1, n_col):
            g.axes[i, j].axis("off")

        # Modify tick sizes
        for j in range(i + 1):
            g.axes[i, j].tick_params(axis="both", which="major", labelsize=tick_fontsize)
            g.axes[i, j].tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Add nice labels
    for i, param_name in enumerate(param_names):
        g.axes[i, 0].set_ylabel(param_name, fontsize=label_fontsize)
        g.axes[len(param_names) - 1, i].set_xlabel(param_name, fontsize=label_fontsize)

    # Add grids
    for i in range(n_params):
        for j in range(n_params):
            g.axes[i, j].grid(alpha=0.5)

    g.tight_layout()
    return g
