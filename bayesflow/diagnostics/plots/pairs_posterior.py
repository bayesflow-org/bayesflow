import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.lines import Line2D

from .pairs_samples import pairs_samples


def pairs_posterior(
    post_samples: np.ndarray,
    prior_samples: np.ndarray = None,
    prior=None,
    variable_names: list = None,
    true_params: np.ndarray = None,
    height: int = 3,
    label_fontsize: int = 14,
    legend_fontsize: int = 16,
    tick_fontsize: int = 12,
    post_color: str | tuple = "#132a70",
    prior_color: str | tuple = "gray",
    post_alpha: float = 0.9,
    prior_alpha: float = 0.7,
    **kwargs,
) -> sns.PairGrid:
    """Generates a bivariate pairplot given posterior draws and optional prior or prior draws.

    post_samples   : np.ndarray of shape (n_post_draws, n_params)
        The posterior draws obtained for a SINGLE observed data set.
    prior_samples       : np.ndarray of shape (n_prior_draws, n_params) or None, optional (default: None)
        The optional prior samples obtained from the prior. If both prior and prior_samples are provided, prior_samples
        will be used.
    prior             : bayesflow.forward_inference.Prior instance or None, optional, default: None
        The optional prior object having an input-output signature as given by bayesflow.forward_inference.Prior
    variable_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    true_params       : np.ndarray of shape (n_params,) or None, optional, default: None
        The true parameter values to be plotted on the diagonal.
    height            : float, optional, default: 3
        The height of the pairplot
    label_fontsize    : int, optional, default: 14
        The font size of the x and y-label texts (parameter names)
    legend_fontsize   : int, optional, default: 16
        The font size of the legend text
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    post_color        : str, optional, default: '#132a70'
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
    context = ""
    g = pairs_samples(
        post_samples, context=context, variable_names=variable_names, render=False, height=height, **kwargs
    )

    # Obtain n_draws and n_params
    n_draws, n_params = post_samples.shape

    # If prior object is given and no draws, obtain draws
    if prior is not None and prior_samples is None:
        draws = prior(n_draws)
        if isinstance(draws, dict):
            prior_samples = draws["prior_draws"]
        else:
            prior_samples = draws
    elif prior_samples is not None:
        # trim to the same number of draws as posterior
        prior_samples = prior_samples[:n_draws]

    # Attempt to determine parameter names
    if variable_names is None:
        if hasattr(prior, "param_names"):
            if prior.variable_names is not None:
                variable_names = prior.variable_names
            else:
                variable_names = [f"{context} $\\theta_{{{i}}}$" for i in range(1, n_params + 1)]
        else:
            variable_names = [f"{context} $\\theta_{{{i}}}$" for i in range(1, n_params + 1)]
    else:
        variable_names = [f"{context} {p}" for p in variable_names]

    # Add prior, if given
    if prior_samples is not None:
        prior_samples_df = pd.DataFrame(prior_samples, columns=variable_names)
        g.data = prior_samples_df
        g.map_diag(sns.histplot, fill=True, color=prior_color, alpha=prior_alpha, kde=True, zorder=-1)
        g.map_lower(sns.kdeplot, fill=True, color=prior_color, alpha=prior_alpha, zorder=-1)

    # Add true parameters
    if true_params is not None:
        # Custom function to plot true_params on the diagonal
        def plot_true_params(x, **kwargs):
            param = x.iloc[0]  # Get the single true value for the diagonal
            plt.axvline(param, color="black", linestyle="--")  # Add vertical line

        # only plot on the diagonal a vertical line for the true parameter
        g.data = pd.DataFrame(true_params[np.newaxis], columns=variable_names)
        g.map_diag(plot_true_params)

    # Add legend, if prior also given
    if prior_samples is not None or prior is not None:
        handles = [
            Line2D(xdata=[], ydata=[], color=post_color, lw=3, alpha=post_alpha),
            Line2D(xdata=[], ydata=[], color=prior_color, lw=3, alpha=prior_alpha),
        ]
        handles_names = ["Posterior", "Prior"]
        if true_params is not None:
            handles.append(Line2D(xdata=[], ydata=[], color="black", lw=3, linestyle="--"))
            handles_names.append("True Parameter")
        plt.legend(handles=handles, labels=handles_names, fontsize=legend_fontsize, loc="center right")

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
    for i, param_name in enumerate(variable_names):
        g.axes[i, 0].set_ylabel(param_name, fontsize=label_fontsize)
        g.axes[len(variable_names) - 1, i].set_xlabel(param_name, fontsize=label_fontsize)

    # Add grids
    for i in range(n_params):
        for j in range(n_params):
            g.axes[i, j].grid(alpha=0.5)

    g.tight_layout()
    return g
