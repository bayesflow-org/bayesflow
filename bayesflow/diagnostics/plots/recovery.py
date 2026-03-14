from collections.abc import Sequence, Mapping, Callable

import matplotlib.pyplot as plt
import numpy as np

from bayesflow.utils import prepare_plot_data, prettify_subplots, make_quadratic, add_titles_and_labels, add_metric
from bayesflow.utils.numpy_utils import credible_interval
from bayesflow.utils.dict_utils import compute_test_quantities
from .plot import CellPlot


def recovery(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    test_quantities: dict[str, Callable] = None,
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
    test_quantities   : dict or None, optional, default: None
        A dict that maps plot titles to functions that compute
        test quantities based on estimate/target draws.

        The dict keys are automatically added to ``variable_keys``
        and ``variable_names``.
        Test quantity functions are expected to accept a dict of draws with
        shape ``(batch_size, ...)`` as the first (typically only)
        positional argument and return an NumPy array of shape
        ``(batch_size,)``.
        The functions do not have to deal with an additional
        sample dimension, as appropriate reshaping is done internally.
    point_agg         : callable, optional, default: median
        Function to compute point estimates.
    uncertainty_agg   : callable, optional, default: credible_interval with coverage probability 95%
        Function to compute a measure of uncertainty. Can either be the lower and upper
        uncertainty bounds provided with the shape (2, num_datasets, num_params) or a
        scalar measure of uncertainty (e.g., the median absolute deviation) with shape
        (num_datasets, num_params).
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
    xlabel            : str, optional, default: "Ground truth"
        The label shown on the x-axis.
    ylabel            : str, optional, default: "Estimate"
        The label shown on the y-axis.
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

    # Optionally, compute and prepend test quantities from draws
    if test_quantities is not None:
        updated_data = compute_test_quantities(
            targets=targets,
            estimates=estimates,
            variable_keys=variable_keys,
            variable_names=variable_names,
            test_quantities=test_quantities,
        )
        variable_names = updated_data["variable_names"]
        variable_keys = updated_data["variable_keys"]
        estimates = updated_data["estimates"]
        targets = updated_data["targets"]

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
        if u.ndim == 3:
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
                yerr=u[..., i],
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


class Recovery(CellPlot):
    """Class-based parameter recovery plot with cell/grid interface.

    For each variable, plots posterior point estimates against ground-truth
    values with uncertainty bars, and optionally annotates with the Pearson
    correlation coefficient.

    Parameters
    ----------
    variable_keys : Sequence[str], optional
        Select keys from the input dictionaries. By default, all keys.
    variable_names : Sequence[str], optional
        Human-readable variable names for subplot titles.
    test_quantities : dict[str, Callable], optional
        Functions to compute test quantities before the main computation.
    figsize : Sequence[float], optional
        Overall figure size.
    num_row : int, optional
        Number of subplot rows. Inferred automatically if not set.
    num_col : int, optional
        Number of subplot columns. Inferred automatically if not set.
    label_fontsize : int, optional (default = 16)
        Font size for axis labels.
    title_fontsize : int, optional (default = 18)
        Font size for subplot titles.
    tick_fontsize : int, optional (default = 12)
        Font size for tick labels.
    point_agg : callable, optional (default = np.median)
        Function to compute the point estimate, called as
        ``point_agg(estimates_i, axis=1)``.
    uncertainty_agg : callable, optional (default = credible_interval)
        Function to compute uncertainty, called as
        ``uncertainty_agg(estimates_i, axis=1)``. Must return either shape
        ``(2, num_datasets)`` for lower/upper bounds or
        ``(num_datasets,)`` for a scalar measure.
    point_agg_kwargs : dict, optional
        Additional keyword arguments forwarded to ``point_agg``.
    uncertainty_agg_kwargs : dict, optional
        Additional keyword arguments forwarded to ``uncertainty_agg``.
    add_corr : bool, optional (default = True)
        Whether to annotate each subplot with the Pearson correlation.
    color : str, optional (default = "#132a70")
        Color for scatter points and error bars.
    markersize : float, optional
        Marker size in points.
    metric_fontsize : int, optional (default = 16)
        Font size for the correlation annotation.
    xlabel : str, optional (default = "Ground truth")
        X-axis label.
    ylabel : str, optional (default = "Estimate")
        Y-axis label.
    """

    def __init__(
        self,
        variable_keys: Sequence[str] = None,
        variable_names: Sequence[str] = None,
        test_quantities: dict[str, Callable] = None,
        figsize: Sequence[float] = None,
        num_row: int = None,
        num_col: int = None,
        label_fontsize: int = 16,
        title_fontsize: int = 18,
        tick_fontsize: int = 12,
        point_agg: Callable = np.median,
        uncertainty_agg: Callable = credible_interval,
        point_agg_kwargs: dict = None,
        uncertainty_agg_kwargs: dict = None,
        add_corr: bool = True,
        color: str = "#132a70",
        markersize: float = None,
        metric_fontsize: int = 16,
        xlabel: str = "Ground truth",
        ylabel: str = "Estimate",
    ):
        super().__init__(
            variable_keys=variable_keys,
            variable_names=variable_names,
            test_quantities=test_quantities,
            figsize=figsize,
            num_row=num_row,
            num_col=num_col,
            label_fontsize=label_fontsize,
            title_fontsize=title_fontsize,
            tick_fontsize=tick_fontsize,
            xlabel=xlabel,
            ylabel=ylabel,
        )
        self.point_agg = point_agg
        self.uncertainty_agg = uncertainty_agg
        self.point_agg_kwargs = point_agg_kwargs or {}
        self.uncertainty_agg_kwargs = uncertainty_agg_kwargs or {}
        self.add_corr = add_corr
        self.color = color
        self.markersize = markersize
        self.metric_fontsize = metric_fontsize

    def plot_cell(
        self,
        ax: plt.Axes,
        estimates_i: np.ndarray,
        targets_i: np.ndarray,
        variable_name: str = None,
        *,
        legend: bool = False,
    ) -> plt.Axes:
        """Draw the recovery scatter for a single variable.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            The axes to draw on.
        estimates_i : np.ndarray of shape (num_datasets, num_draws)
            Posterior draws for one variable.
        targets_i : np.ndarray of shape (num_datasets,)
            Ground-truth values for one variable.
        variable_name : str, optional
            Name of the variable, used as the subplot title.
        legend : bool, optional (default = False)
            Unused; included for interface consistency.

        Returns
        -------
        matplotlib.axes.Axes
        """
        point_estimate = self.point_agg(estimates_i, axis=1, **self.point_agg_kwargs)

        if self.uncertainty_agg is not None:
            u = self.uncertainty_agg(estimates_i, axis=1, **self.uncertainty_agg_kwargs)
            if u.ndim == 2:
                # u has shape (2, num_datasets): lower and upper bounds
                yerr = np.array([point_estimate - u[0], u[1] - point_estimate])
            else:
                # u has shape (num_datasets,): scalar uncertainty measure
                yerr = u
            ax.errorbar(
                targets_i,
                point_estimate,
                yerr=yerr,
                fmt="o",
                alpha=0.5,
                color=self.color,
                markersize=self.markersize,
            )
        else:
            ax.scatter(
                targets_i,
                point_estimate,
                alpha=0.5,
                color=self.color,
                s=None if self.markersize is None else self.markersize**2,
            )

        make_quadratic(ax, targets_i, point_estimate)

        if self.add_corr:
            corr = np.corrcoef(targets_i, point_estimate)[0, 1]
            add_metric(ax=ax, metric_text="$r$", metric_value=corr, metric_fontsize=self.metric_fontsize)

        if variable_name is not None:
            ax.set_title(variable_name, fontsize=self.title_fontsize)

        return ax
