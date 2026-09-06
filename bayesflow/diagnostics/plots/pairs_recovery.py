from collections.abc import Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from bayesflow.utils.dict_utils import compute_test_quantities, dicts_to_arrays
from bayesflow.utils.exceptions import ShapeError
from bayesflow.utils.numpy_utils import credible_interval
from bayesflow.utils.plot_utils import add_metric, make_quadratic, prettify_subplots
from bayesflow.utils.validators import check_estimates_prior_shapes


def pairs_recovery(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] | str = None,
    variable_names: Sequence[str] | str = None,
    test_quantities: dict[str, Callable] = None,
    point_agg: Callable = np.median,
    uncertainty_agg: Callable = credible_interval,
    point_agg_kwargs: dict = None,
    uncertainty_agg_kwargs: dict = None,
    add_corr: bool = True,
    height: float = 2.5,
    color: str = "#132a70",
    alpha: float = 0.5,
    label_fontsize: int = 14,
    tick_fontsize: int = 12,
    metric_fontsize: int = 14,
    markersize: float = None,
) -> plt.Figure:
    """Plot recovery and pairwise dependencies across simulated datasets.

    Diagonal cells show ground truth (x) versus point estimate (y), with
    optional uncertainty intervals and an identity line. Upper cells show
    pairs of point estimates; lower cells show pairs of ground truths.
    In cell (i, j), the x variable is j and the y variable is i. Each point
    represents one dataset, including in the off-diagonal cells.

    Parameters
    ----------
    estimates : np.ndarray or dict[str, np.ndarray]
        Posterior draws of shape (num_datasets, num_draws, num_variables),
        or a dictionary of arrays with these leading dataset/draw axes.
    targets : np.ndarray or dict[str, np.ndarray]
        Corresponding truths of shape (num_datasets, num_variables), or a
        dictionary matching the selected estimate keys.
    variable_keys : sequence of str or str, optional
        Dictionary keys to select, in plotting order. Defaults to all keys
        in estimates.
    variable_names : sequence of str or str, optional
        Display names for the selected scalar variables. Inferred if omitted.
    test_quantities : dict[str, Callable], optional
        Named quantities to prepend. Requires dictionary inputs. Each callable
        accepts ``data=`` containing a dictionary of draws with a leading batch
        axis and returns one scalar per batch element. Posterior dataset and
        draw axes are flattened and restored by ``compute_test_quantities``.
    point_agg : Callable, default np.median
        Called with ``axis=1`` to produce (num_datasets, num_variables) points.
    uncertainty_agg : Callable or None, default credible_interval
        Called with ``axis=1``. Returns nonnegative symmetric errors of shape
        (num_datasets, num_variables), or lower/upper bounds of shape
        (2, num_datasets, num_variables) enclosing the point estimates.
        Use None to omit error bars.
    point_agg_kwargs : dict, optional
        Additional arguments for point_agg.
    uncertainty_agg_kwargs : dict, optional
        Additional arguments for uncertainty_agg, e.g., ``dict(prob=0.5)``
        for a 50% credible interval instead of the default 95% interval.
    add_corr : bool, default True
        Annotate only diagonal cells with truth/estimate Pearson correlation.
    height : float, default 2.5
        Figure width and height per variable, in inches.
    color : str, default "#132a70"
        Color of points and error bars.
    alpha : float, default 0.5
        Opacity of points and error bars.
    label_fontsize : int, default 14
        Font size of variable and source labels on every cell.
    tick_fontsize : int, default 12
        Font size of tick labels.
    metric_fontsize : int, default 14
        Font size of correlation annotations.
    markersize : float, optional
        Marker size in points, as in recovery; squared for scatter areas.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with num_variables squared axes in row-major order. Axes are
        independent because the source changes across each row and column.
        Diagonal cells have equal x/y limits including uncertainty bounds.

    Raises
    ------
    ShapeError
        If selected inputs are empty, have incompatible shapes, or aggregators
        return shapes other than those documented above.
    TypeError
        If test quantities are requested with array inputs.
    ValueError
        If variable names do not match the selected variables, errors are
        negative, or uncertainty bounds do not enclose the point estimates.

    Notes
    -----
    Pairwise estimates describe variation of posterior point summaries across
    datasets, not posterior dependence within a dataset. Posterior summaries
    are heuristic and can be misleading for multimodal distributions.
    """
    if test_quantities is not None:
        if not isinstance(estimates, Mapping) or not isinstance(targets, Mapping):
            raise TypeError("test_quantities requires dictionaries for estimates and targets.")
        if variable_keys is not None:
            variable_keys = [variable_keys] if isinstance(variable_keys, str) else list(variable_keys)
        if variable_names is not None:
            variable_names = [variable_names] if isinstance(variable_names, str) else list(variable_names)
        updated = compute_test_quantities(
            estimates=estimates,
            targets=targets,
            variable_keys=variable_keys,
            variable_names=variable_names,
            test_quantities=test_quantities,
        )
        estimates, targets = updated["estimates"], updated["targets"]
        variable_keys, variable_names = updated["variable_keys"], updated["variable_names"]

    data = dicts_to_arrays(
        estimates=estimates,
        targets=targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
    )
    estimates, targets = data["estimates"], data["targets"]
    check_estimates_prior_shapes(estimates, targets)
    if estimates.ndim != 3 or any(size == 0 for size in estimates.shape):
        raise ShapeError("estimates must have nonempty dataset, draw, and variable axes.")
    variable_names = estimates.variable_names
    points = np.asarray(point_agg(estimates, axis=1, **(point_agg_kwargs or {})))
    if points.shape != targets.shape:
        raise ShapeError("point_agg must return shape (num_datasets, num_variables).")
    errors = None
    if uncertainty_agg is not None:
        uncertainty = np.asarray(uncertainty_agg(estimates, axis=1, **(uncertainty_agg_kwargs or {})))
        if uncertainty.shape == (2, *points.shape):
            # Do not modify a caller-owned (possibly read-only) bounds array.
            errors = np.stack((points - uncertainty[0], uncertainty[1] - points))
        elif uncertainty.shape == points.shape:
            errors = np.stack((uncertainty, uncertainty))
        else:
            raise ShapeError("uncertainty_agg must return shape (num_datasets, num_variables) or (2, ...).")
        if np.any(errors < 0):
            raise ValueError("Uncertainty errors must be nonnegative and bounds must enclose the point estimates.")

    n = points.shape[-1]
    fig, axes = plt.subplots(n, n, figsize=(height * n, height * n), squeeze=False, layout="constrained")
    for i in range(n):
        for j in range(n):
            ax = axes[i, j]
            x = targets[:, j] if i >= j else points[:, j]
            y = targets[:, i] if i > j else points[:, i]
            if i == j and errors is not None:
                ax.errorbar(x, y, yerr=errors[..., i], fmt="o", color=color, alpha=alpha, markersize=markersize)
            else:
                ax.scatter(x, y, color=color, alpha=alpha, s=None if markersize is None else markersize**2)
            if i == j:
                extent = y if errors is None else np.concatenate((y, y - errors[0, :, i], y + errors[1, :, i]))
                make_quadratic(ax, x, extent)
                if add_corr:
                    add_metric(ax, "$r$", np.corrcoef(x, y)[0, 1], metric_fontsize=metric_fontsize)
            ax.set_box_aspect(1)
            ax.set_xlabel(f"{variable_names[j]}\n({'Ground truth' if i >= j else 'Estimate'})", fontsize=label_fontsize)
            ax.set_ylabel(f"{variable_names[i]}\n({'Ground truth' if i > j else 'Estimate'})", fontsize=label_fontsize)
    prettify_subplots(axes, num_subplots=n * n, tick_fontsize=tick_fontsize)
    return fig
