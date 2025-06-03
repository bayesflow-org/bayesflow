from collections.abc import Sequence, Mapping

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from bayesflow.utils.dict_utils import dicts_to_arrays
from bayesflow.utils.plot_utils import create_legends

from .pairs_samples import _pairs_samples


def pairs_posterior(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray = None,
    priors: Mapping[str, np.ndarray] | np.ndarray = None,
    dataset_id: int = None,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    height: int = 3,
    post_color: str | tuple = "#132a70",
    prior_color: str | tuple = "gray",
    target_color: str | tuple = "red",
    alpha: float = 0.9,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    legend_fontsize: int = 14,
    **kwargs,
) -> sns.PairGrid:
    """Generates a bivariate pair plot given posterior draws and optional prior or prior draws.

    Parameters
    ----------
    estimates   : np.ndarray of shape (n_post_draws, n_params)
        The posterior draws obtained for a SINGLE observed data set.
    targets       : np.ndarray of shape (n_params,) or None, optional, default: None
        Optional true parameter values that have generated the observed dataset.
    priors       : np.ndarray of shape (n_prior_draws, n_params) or None, optional (default: None)
        Optional prior samples obtained from the prior.
    dataset_id: Optional ID of the dataset for whose posterior the pair plots shall be generated.
        Should only be specified if estimates contain posterior draws from multiple datasets.
    variable_keys       : list or None, optional, default: None
       Select keys from the dictionary provided in samples.
       By default, select all keys.
    variable_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    height            : float, optional, default: 3
        The height of the pair plots
    label_fontsize    : int, optional, default: 14
        The font size of the x and y-label texts (parameter names)
    tick_fontsize     : int, optional, default: 12
        The font size of the axis tick labels
    legend_fontsize   : int, optional, default: 16
        The font size of the legend text
    post_color        : str, optional, default: '#132a70'
        The color for the posterior histograms and KDEs
    prior_color      : str, optional, default: gray
        The color for the optional prior histograms and KDEs
    target_color      : str, optional, default: red
        The color for the optional true parameter lines and points
    alpha             : float in [0, 1], optional, default: 0.9
        The opacity of the posterior plots

    **kwargs          : dict, optional, default: {}
        Further optional keyword arguments propagated to `_pairs_samples`

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    AssertionError
        If the shape of posterior_draws is not 2-dimensional.
    """

    plot_data = dicts_to_arrays(
        estimates=estimates,
        targets=targets,
        priors=priors,
        dataset_ids=dataset_id,
        variable_keys=variable_keys,
        variable_names=variable_names,
    )

    # dicts_to_arrays will keep the dataset axis even if it is of length 1
    # however, pairs plotting requires the dataset axis to be removed
    estimates_shape = plot_data["estimates"].shape
    if len(estimates_shape) == 3 and estimates_shape[0] == 1:
        plot_data["estimates"] = np.squeeze(plot_data["estimates"], axis=0)

    # plot posterior first
    g = _pairs_samples(
        plot_data=plot_data,
        height=height,
        color=post_color,
        color2=prior_color,
        alpha=alpha,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
        legend_fontsize=legend_fontsize,
        **kwargs,
    )

    targets = plot_data.get("targets")
    if targets is not None:
        # Ensure targets is at least 2D
        if targets.ndim == 1:
            targets = np.atleast_2d(targets)

        # Create DataFrame with variable names as columns
        g.data = pd.DataFrame(targets, columns=targets.variable_names)
        g.data["_source"] = "True Parameter"
        g.map_diag(plot_true_params_as_lines, color=target_color)
        g.map_offdiag(plot_true_params_as_points, color=target_color)

        create_legends(
            g,
            plot_data,
            color=post_color,
            color2=prior_color,
            legend_fontsize=legend_fontsize,
            show_single_legend=False,
        )

    return g


def plot_true_params_as_lines(x, hue=None, color=None, **kwargs):
    """Custom function to plot true parameters on the diagonal as dashed lines."""
    # hue needs to be added to handle the case of plotting both posterior and prior
    param = x.iloc[0]  # Get the single true value for the diagonal
    # only plot on the diagonal a vertical line for the true parameter
    plt.axvline(param, color=color, linestyle="--")


def plot_true_params_as_points(x, y, color=None, marker="x", **kwargs):
    """Custom function to plot true parameters on the off-diagonal as a single point."""
    if len(x) > 0 and len(y) > 0:
        plt.scatter(x.iloc[0], y.iloc[0], color=color, marker=marker, **kwargs)
