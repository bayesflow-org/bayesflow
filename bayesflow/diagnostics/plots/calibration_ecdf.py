from collections.abc import Callable, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt

from ...utils.dict_utils import compute_test_quantities
from ...utils.plot_utils import prepare_plot_data, add_titles_and_labels, prettify_subplots
from ...utils.ecdf import simultaneous_ecdf_bands
from ...utils.ecdf.ranks import fractional_ranks, distance_ranks
from .plot import CellPlot


def calibration_ecdf(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    test_quantities: dict[str, Callable] = None,
    difference: bool = True,
    stacked: bool = False,
    rank_type: str | np.ndarray = "fractional",
    figsize: Sequence[float] = None,
    label_fontsize: int = 16,
    legend_fontsize: int = 14,
    legend_location: str = "lower right",
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    rank_ecdf_color: str = "#132a70",
    fill_color: str = "grey",
    num_row: int = None,
    num_col: int = None,
    **kwargs,
) -> plt.Figure:
    """
    Creates the empirical CDFs for each marginal rank distribution
    and plots it against a uniform ECDF.
    ECDF simultaneous bands are drawn using simulations from the uniform,
    as proposed by [1].

    For models with many parameters, use `stacked=True` to obtain an idea
    of the overall calibration of a posterior approximator.

    To compute ranks based on the Euclidean distance to the origin or a reference, use `rank_type='distance'` (and
    pass a reference array, respectively). This can be used to check the joint calibration of the posterior approximator
    and might show potential biases in the posterior approximation which are not detected by the fractional ranks (e.g.,
    when the prior equals the posterior). This is motivated by [2].

    [1] Säilynoja, T., Bürkner, P. C., & Vehtari, A. (2022). Graphical test
    for discrete uniformity and its applications in goodness-of-fit evaluation
    and multiple sample comparison. Statistics and Computing, 32(2), 1-21.
    https://arxiv.org/abs/2103.10522

    [2] Lemos, Pablo, et al. "Sampling-based accuracy testing of posterior estimators
     for general inference." International Conference on Machine Learning. PMLR, 2023.
     https://proceedings.mlr.press/v202/lemos23a.html

    Parameters
    ----------
    estimates      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    targets     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets
    difference        : bool, optional, default: True
        If `True`, plots the ECDF difference.
        Enables a more dynamic visualization range.
    stacked           : bool, optional, default: False
        If `True`, all ECDFs will be plotted on the same plot.
        If `False`, each ECDF will have its own subplot,
        similar to the behavior of `calibration_histogram`.
    rank_type   : str, optional, default: 'fractional'
        If `fractional` (default), the ranks are computed as the fraction
        of posterior samples that are smaller than the prior.
        If `distance`, the ranks are computed as the fraction of posterior
        samples that are closer to a reference points (default here is the origin).
        You can pass a reference array in the same shape as the
        `estimates` array by setting `targets` in the ``ranks_kwargs``.
        This is motivated by [2].
    variable_keys       : list or None, optional, default: None
       Select keys from the dictionaries provided in estimates and targets.
       By default, select all keys.
    variable_names    : list or None, optional, default: None
        The parameter names for nice plot titles.
        Inferred if None. Only relevant if `stacked=False`.
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
    figsize           : tuple or None, optional, default: None
        The figure size passed to the matplotlib constructor.
        Inferred if None.
    label_fontsize    : int, optional, default: 16
        The font size of the y-label and y-label texts
    legend_fontsize   : int, optional, default: 14
        The font size of the legend text.
    legend_location : str, optional, default: 'lower right
        The location of the legend.
    title_fontsize    : int, optional, default: 18
        The font size of the title text.
        Only relevant if `stacked=False`
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    rank_ecdf_color   : str, optional, default: '#a34f4f'
        The color to use for the rank ECDFs
    fill_color        : str, optional, default: 'grey'
        The color of the fill arguments.
    num_row           : int, optional, default: None
        The number of rows for the subplots.
        Dynamically determined if None.
    num_col           : int, optional, default: None
        The number of columns for the subplots.
        Dynamically determined if None.
    **kwargs          : dict, optional, default: {}
        Keyword arguments can be passed to control the behavior of
        ECDF simultaneous band computation through the ``ecdf_bands_kwargs``
        dictionary. See `simultaneous_ecdf_bands` for keyword arguments.
        Moreover, additional keyword arguments can be passed to control the behavior of
        the rank computation through the ``ranks_kwargs`` dictionary.

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of `estimates`
        and `targets`.
    ValueError
        If an unknown `rank_type` is passed.
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

    plot_data = prepare_plot_data(
        estimates=estimates,
        targets=targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
        num_col=num_col,
        num_row=num_row,
        figsize=figsize,
        stacked=stacked,
    )

    estimates = plot_data.pop("estimates")
    targets = plot_data.pop("targets")

    if rank_type == "fractional":
        # Compute fractional ranks
        ranks = fractional_ranks(estimates, targets)
    elif rank_type == "distance":
        # Compute ranks based on distance to the origin
        ranks = distance_ranks(estimates, targets, stacked=stacked, **kwargs.pop("ranks_kwargs", {}))
    else:
        raise ValueError(f"Unknown rank type: {rank_type}. Use 'fractional' or 'distance'.")

    # Plot individual ecdf of parameters
    for j in range(ranks.shape[-1]):
        xx = np.repeat(np.sort(ranks[:, j]), 2)
        xx = np.pad(xx, (1, 1), constant_values=(0, 1))
        yy = np.linspace(0, 1, num=xx.shape[-1] // 2)
        yy = np.repeat(yy, 2)

        # Difference, if specified
        if difference:
            yy -= xx

        if stacked:
            if j == 0:
                if not isinstance(plot_data["axes"], np.ndarray):
                    plot_data["axes"] = np.array([plot_data["axes"]])  # in case of single axis
                plot_data["axes"][0].plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label="Rank ECDFs")
            else:
                plot_data["axes"][0].plot(xx, yy, color=rank_ecdf_color, alpha=0.95)
        else:
            plot_data["axes"].flat[j].plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label="Rank ECDF")

    # Compute uniform ECDF and bands
    alpha, z, L, U = simultaneous_ecdf_bands(estimates.shape[0], **kwargs.pop("ecdf_bands_kwargs", {}))

    # Difference, if specified
    if difference:
        L -= z
        U -= z
        ylab = "ECDF Difference"
    else:
        ylab = "ECDF"

    # Add simultaneous bounds
    if not stacked:
        titles = plot_data["variable_names"]
    elif rank_type in ["distance", "random"]:
        titles = ["Joint ECDFs"]
    else:
        titles = ["Stacked ECDFs"]

    for i, (ax, title) in enumerate(zip(plot_data["axes"].flat, titles)):
        ax.fill_between(z, L, U, color=fill_color, alpha=0.2, label=rf"{int((1 - alpha) * 100)}$\%$ Confidence Bands")
        ax.set_title(title, fontsize=title_fontsize)

        if i == 0:
            ax.legend(fontsize=legend_fontsize, loc=legend_location)

    prettify_subplots(plot_data["axes"], num_subplots=plot_data["num_variables"], tick_fontsize=tick_fontsize)

    add_titles_and_labels(
        plot_data["axes"],
        plot_data["num_row"],
        plot_data["num_col"],
        xlabel=f"{rank_type.capitalize()} rank statistic",
        ylabel=ylab,
        label_fontsize=label_fontsize,
    )

    plot_data["fig"].tight_layout()
    return plot_data["fig"]


class CalibrationECDF(CellPlot):
    """Class-based calibration ECDF plot with cell/grid interface.

    Plots the empirical CDF of rank statistics for each variable as a subplot
    grid, with simultaneous confidence bands as proposed by [1].

    [1] Säilynoja, T., Bürkner, P. C., & Vehtari, A. (2022). Graphical test
    for discrete uniformity and its applications in goodness-of-fit evaluation
    and multiple sample comparison. Statistics and Computing, 32(2), 1-21.
    https://arxiv.org/abs/2103.10522

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
    difference : bool, optional (default = True)
        If True, plots the ECDF difference (ECDF minus uniform), enabling a
        more dynamic visualization range.
    rank_type : str, optional (default = "fractional")
        ``"fractional"`` computes fractional ranks; ``"distance"`` computes
        ranks based on Euclidean distance to the origin (or a reference).
    rank_ecdf_color : str, optional (default = "#132a70")
        Color for the rank ECDF lines.
    fill_color : str, optional (default = "grey")
        Color for the simultaneous confidence band fill.
    legend_fontsize : int, optional (default = 14)
        Font size for the legend (shown on the first subplot only).
    legend_location : str, optional (default = "lower right")
        Location of the legend.
    ecdf_bands_kwargs : dict, optional
        Additional keyword arguments forwarded to
        :func:`simultaneous_ecdf_bands`.
    ranks_kwargs : dict, optional
        Additional keyword arguments forwarded to
        :func:`distance_ranks` when ``rank_type="distance"``.
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
        difference: bool = True,
        rank_type: str = "fractional",
        rank_ecdf_color: str = "#132a70",
        fill_color: str = "grey",
        legend_fontsize: int = 14,
        legend_location: str = "lower right",
        ecdf_bands_kwargs: dict = None,
        ranks_kwargs: dict = None,
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
            xlabel=f"{rank_type.capitalize()} rank statistic",
            ylabel="ECDF Difference" if difference else "ECDF",
        )
        self.difference = difference
        self.rank_type = rank_type
        self.rank_ecdf_color = rank_ecdf_color
        self.fill_color = fill_color
        self.legend_fontsize = legend_fontsize
        self.legend_location = legend_location
        self.ecdf_bands_kwargs = ecdf_bands_kwargs or {}
        self.ranks_kwargs = ranks_kwargs or {}

    def plot_cell(
        self,
        ax: plt.Axes,
        estimates_i: np.ndarray,
        targets_i: np.ndarray,
        variable_name: str = None,
        *,
        legend: bool = False,
    ) -> plt.Axes:
        """Draw the calibration ECDF for a single variable.

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
            Whether to add a legend to this cell.

        Returns
        -------
        matplotlib.axes.Axes
        """
        if self.rank_type == "fractional":
            ranks_i = np.mean(estimates_i < targets_i[:, np.newaxis], axis=1)
        elif self.rank_type == "distance":
            ranks_i = distance_ranks(
                estimates_i[:, :, np.newaxis],
                targets_i[:, np.newaxis],
                stacked=False,
                **self.ranks_kwargs,
            )[:, 0]
        else:
            raise ValueError(f"Unknown rank type: {self.rank_type}. Use 'fractional' or 'distance'.")

        # Build step-function ECDF
        xx = np.repeat(np.sort(ranks_i), 2)
        xx = np.pad(xx, (1, 1), constant_values=(0, 1))
        yy = np.linspace(0, 1, num=xx.shape[-1] // 2)
        yy = np.repeat(yy, 2)

        if self.difference:
            yy -= xx

        ax.plot(xx, yy, color=self.rank_ecdf_color, alpha=0.95, label="Rank ECDF")

        # Simultaneous ECDF bands (depend only on num_datasets)
        alpha, z, L, U = simultaneous_ecdf_bands(estimates_i.shape[0], **self.ecdf_bands_kwargs)

        if self.difference:
            L -= z
            U -= z

        ax.fill_between(
            z,
            L,
            U,
            color=self.fill_color,
            alpha=0.2,
            label=rf"{int((1 - alpha) * 100)}$\%$ Confidence Bands",
        )

        if variable_name is not None:
            ax.set_title(variable_name, fontsize=self.title_fontsize)

        if legend:
            ax.legend(fontsize=self.legend_fontsize, loc=self.legend_location)

        return ax
