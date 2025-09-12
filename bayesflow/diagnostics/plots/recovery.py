from collections.abc import Sequence, Mapping, Callable

import matplotlib.pyplot as plt
import numpy as np

from bayesflow.utils import prepare_plot_data, prettify_subplots, make_quadratic, add_titles_and_labels, add_metric
from bayesflow.utils.numpy_utils import credible_interval


def recovery(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    point_agg: Callable = np.median,
    uncertainty_agg: Callable = credible_interval,
    point_agg_kwargs: dict = None,
    uncertainty_agg_kwargs: dict = None,
    add_corr: bool = True,
    figsize: Sequence[int] = None,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    metric_fontsize: int = 16,
    tick_fontsize: int = 12,
    color: str = "#132a70",
    num_col: int = None,
    num_row: int = None,
    xlabel: str = "Ground truth",
    ylabel: str = "Estimate",
    markersize: float = None,
    **kwargs,
) -> plt.Figure:
    """
    Creates and plots publication-ready recovery plot with true estimate
    vs. point estimate + uncertainty.
    The point estimate can be controlled with the ``point_agg`` argument,
    and the uncertainty estimate can be controlled with the
    ``uncertainty_agg`` argument.

    This plot yields similar information as the "posterior z-score",
    but allows for generic point and uncertainty estimates:

    https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html

    Important:
    Posterior aggregates play no special role in Bayesian inference and should only be used heuristically.
    For instance, in the case of multi-modal posteriors, common point estimates, such as mean, (geometric) median,
    or maximum a posteriori (MAP) mean nothing.

    Parameters
    ----------
    estimates           : np.ndarray of shape (num_datasets, num_post_draws, num_params)
        The posterior draws obtained from num_datasets
    targets        : np.ndarray of shape (num_datasets, num_params)
        The prior draws (true parameters) used for generating the num_datasets
    variable_keys       : list or None, optional, default: None
       Select keys from the dictionaries provided in estimates and targets.
       By default, select all keys.
    variable_names    : list or None, optional, default: None
        The individual parameter names for nice plot titles. Inferred if None
    point_agg         : callable, optional, default: median
        Function to compute point estimates.
    uncertainty_agg   : callable, optional, default: credible_interval with coverage probability 95%
        Function to compute uncertainty interval bounds.
    point_agg_kwargs : Optional dictionary of further arguments passed to point_agg.
    uncertainty_agg_kwargs : Optional dictionary of further arguments passed to uncertainty_agg.
        For example, to change the coverage probability of credible_interval to 50%,
        use uncertainty_agg_kwargs = dict(prob=0.5)
    add_corr          : boolean, default: True
        Should correlations between estimates and ground truth values be shown?
    figsize           : tuple or None, optional, default : None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 16
        The font size of the y-label text.
    title_fontsize    : int, optional, default: 18
        The font size of the title text.
    metric_fontsize   : int, optional, default: 16
        The font size of the metrics shown as text.
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels.
    color             : str, optional, default: '#8f2727'
        The color for the true vs. estimated scatter points and error bars.
    num_row           : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    num_col           : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.
    xlabel :
    ylabel :
    markersize        : float, optional, default: None
        The marker size in points.

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation from the expected shapes of ``estimates`` and ``targets``.
    """

    # Gather plot data and metadata into a dictionary
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

    point_agg_kwargs = point_agg_kwargs or {}
    uncertainty_agg_kwargs = uncertainty_agg_kwargs or {}

    # Compute point estimates and uncertainties
    point_estimate = point_agg(estimates, axis=1, **point_agg_kwargs)

    if uncertainty_agg is not None:
        u = uncertainty_agg(estimates, axis=1, **uncertainty_agg_kwargs)
        # compute lower and upper error
        u[0, :, :] = point_estimate - u[0, :, :]
        u[1, :, :] = u[1, :, :] - point_estimate

    for i, ax in enumerate(plot_data["axes"].flat):
        if i >= plot_data["num_variables"]:
            break

        # Add scatter and error bars
        if uncertainty_agg is not None:
            _ = ax.errorbar(
                targets[:, i],
                point_estimate[:, i],
                yerr=u[:, :, i],
                fmt="o",
                alpha=0.5,
                color=color,
                markersize=markersize,
                **kwargs,
            )
        else:
            _ = ax.scatter(
                targets[:, i],
                point_estimate[:, i],
                alpha=0.5,
                color=color,
                s=None if markersize is None else markersize**2,
                **kwargs,
            )

        make_quadratic(ax, targets[:, i], point_estimate[:, i])

        if add_corr:
            corr = np.corrcoef(targets[:, i], point_estimate[:, i])[0, 1]
            add_metric(ax=ax, metric_text="$r$", metric_value=corr, metric_fontsize=metric_fontsize)

        ax.set_title(plot_data["variable_names"][i], fontsize=title_fontsize)

    # Add custom schmuck
    prettify_subplots(plot_data["axes"], num_subplots=plot_data["num_variables"], tick_fontsize=tick_fontsize)
    add_titles_and_labels(
        axes=plot_data["axes"],
        num_row=plot_data["num_row"],
        num_col=plot_data["num_col"],
        xlabel=xlabel,
        ylabel=ylabel,
        label_fontsize=label_fontsize,
    )

    plot_data["fig"].tight_layout()
    return plot_data["fig"]
