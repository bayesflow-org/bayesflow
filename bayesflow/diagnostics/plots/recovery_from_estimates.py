from collections.abc import Callable, Sequence, Mapping

import matplotlib.pyplot as plt
import numpy as np


from bayesflow.utils import prepare_plot_data, prettify_subplots, make_quadratic, add_titles_and_labels, add_metric


def recovery_from_estimates(
    estimates: Mapping[str, Mapping[str, np.ndarray]],
    targets: Mapping[str, np.ndarray],
    marker_mapping: dict[str, str],
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    add_corr: bool = True,
    corr_point_agg: Callable = np.median,
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
    **kwargs,
) -> plt.Figure:
    """
    Creates and plots publication-ready recovery plot of estimates vs. targets.

    This plot yields similar information as the "posterior z-score",
    but allows for generic point and uncertainty estimates:

    https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html

    Important:
    Posterior aggregates play no special role in Bayesian inference and should only be used heuristically.
    For instance, in the case of multi-modal posteriors, common point estimates, such as mean, (geometric) median,
    or maximum a posteriori (MAP) need to be interpreted carefully.

    Parameters
    ----------
    estimates         : dict[str, dict[str, np.ndarray]]
        The model-generated estimates in a nested dictionary, e.g. as returned
        by approximator.estimate(conditions=...).
        - The outer keys identify the inference variable.
        - The inner keys identify point estimates.
        - The inner value is an ndarray of shape (num_datasets, point_estimate_size, variable_block_size)
    targets           : dict[str, np.ndarray]
        The prior draws (true parameters) used for generating the num_datasets
    marker_mapping    : dict[str, str]
        Define how to mark different point estimates by their key, e.g. {"quantiles":"_", "mean":"*"}.
        Only point estimates whose key appears in the dictionary, will be plotted.
    variable_keys     : list or None, optional, default: None
       Select keys from the dictionaries provided in estimates and targets.
       By default, select all keys.
    variable_names    : list or None, optional, default: None
        The individual parameter names for nice plot titles. Inferred if None
    add_corr          : boolean, default: True
        Should correlations between estimates and ground truth values be shown?
    corr_point_agg    : Callable
        Function producing a central point estimate from the whole list of point estimates
        in case correlations should be computed. Default: median
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
    xlabel:
    ylabel:

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation from the expected shapes of ``estimates`` and ``targets``.
    """

    point_estimates = {}
    markers = []
    for var_key, var_value in estimates.items():
        estimate_list = []
        for estimate_key in marker_mapping.keys():
            e = estimates[var_key][estimate_key]
            # move axis of flexible shape to the end
            e = np.moveaxis(e, 1, -1)
            # add dimensions of size (1,) if not 3D yet
            e = np.atleast_3d(e)
            # undo reordering
            e = np.moveaxis(e, -1, 1)
            estimate_list.append(e)
            markers += [marker_mapping[estimate_key]] * e.shape[1]
        point_estimates[var_key] = np.concatenate(estimate_list, axis=1)

    # Gather plot data and metadata into a dictionary
    plot_data = prepare_plot_data(
        estimates=point_estimates,
        targets=targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
        num_col=num_col,
        num_row=num_row,
        figsize=figsize,
    )

    estimates = plot_data.pop("estimates")
    targets = plot_data.pop("targets")

    # Compute point estimates and uncertainties
    point_estimate = corr_point_agg(estimates, axis=1)

    num_quantiles = estimates.shape[1]

    for i, ax in enumerate(plot_data["axes"].flat):
        if i >= plot_data["num_variables"]:
            break

        # Add scatter and error bars
        for q_idx in range(num_quantiles):
            _ = ax.scatter(
                targets[:, i],
                estimates[:, q_idx, i],
                marker=markers[q_idx],
                alpha=0.5,
                color=color,
                **kwargs,
            )

        connecting_lines_x = np.tile(targets[:, i], (2, 1))
        connecting_lines_y = np.array([estimates[:, :, i].min(axis=1), estimates[:, :, i].max(axis=1)])
        _ = ax.plot(
            connecting_lines_x,
            connecting_lines_y,
            alpha=0.5,
            color=color,
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
