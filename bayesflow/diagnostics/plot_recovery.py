from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import median_abs_deviation

from bayesflow.utils.plot_utils import preprocess, prettify_subplots, add_labels, make_quadratic


def plot_recovery(
    post_samples: dict[str, np.ndarray] | np.ndarray,
    prior_samples: dict[str, np.ndarray] | np.ndarray,
    names: Sequence[str] = None,
    point_agg: callable = np.median,
    uncertainty_agg: callable = median_abs_deviation,
    figsize: Sequence[int] = None,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    metric_fontsize: int = 16,
    tick_fontsize: int = 12,
    add_corr: bool = True,
    color: str = "#1a1e75",
    num_col: int = None,
    num_row: int = None,
    xlabel: str = "Ground truth",
    ylabel: str = "Estimate",
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

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation from the expected shapes of ``post_samples`` and ``prior_samples``.
    """

    # Gather plot data and metadata into a dictionary
    plot_data = preprocess(post_samples, prior_samples, names, num_col, num_row, figsize)

    # Compute point estimates and uncertainties
    point_estimate = point_agg(plot_data["post_samples"], axis=1)

    if uncertainty_agg is not None:
        u = uncertainty_agg(plot_data["post_samples"], axis=1)

    for i, ax in enumerate(plot_data["axes"].flat):
        if i >= plot_data["num_variables"]:
            break

        # Add scatter and error bars
        if uncertainty_agg is not None:
            _ = ax.errorbar(
                plot_data["prior_samples"][:, i],
                point_estimate[:, i],
                yerr=u[:, i],
                fmt="o",
                alpha=0.5,
                color=color,
                **kwargs,
            )
        else:
            _ = ax.scatter(plot_data["prior_samples"][:, i], point_estimate[:, i], alpha=0.5, color=color, **kwargs)

        make_quadratic(ax, plot_data["prior_samples"][:, i], point_estimate[:, i])

        # TODO - Generalize
        if add_corr:
            corr = np.corrcoef(plot_data["prior_samples"][:, i], point_estimate[:, i])[0, 1]
            ax.text(
                0.1,
                0.8,
                "$r$ = {:.3f}".format(corr),
                horizontalalignment="left",
                verticalalignment="center",
                transform=ax.transAxes,
                size=metric_fontsize,
            )

        ax.set_title(plot_data["names"][i], fontsize=title_fontsize)

    # Add custom schmuck
    prettify_subplots(plot_data["axes"], num_subplots=plot_data["num_variables"], tick_fontsize=tick_fontsize)
    add_labels(plot_data["axes"], plot_data["num_row"], plot_data["num_col"], xlabel, ylabel, label_fontsize)

    plot_data["fig"].tight_layout()
    return plot_data["fig"]
