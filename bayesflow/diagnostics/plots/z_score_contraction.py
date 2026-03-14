from collections.abc import Callable, Sequence, Mapping

import matplotlib.pyplot as plt
import numpy as np

from bayesflow.utils import prepare_plot_data, add_titles_and_labels, prettify_subplots
from bayesflow.utils.dict_utils import compute_test_quantities
from .plot import CellPlot


def z_score_contraction(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    test_quantities: dict[str, Callable] = None,
    figsize: Sequence[int] = None,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    color: str = "#132a70",
    num_col: int = None,
    num_row: int = None,
    markersize: float = None,
) -> plt.Figure:
    """
    Implements a graphical check for global model sensitivity by plotting the
    posterior z-score over the posterior contraction for each set of posterior
    samples in ``estimates`` according to [1].

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
    estimates      : np.ndarray of shape (num_datasets, num_post_draws, num_params)
        The posterior draws obtained from num_datasets
    targets     : np.ndarray of shape (num_datasets, num_params)
        The prior draws (true parameters) used for generating the num_datasets
    variable_keys       : list or None, optional, default: None
       Select keys from the dictionaries provided in estimates and targets.
       By default, select all keys.
    variable_names    : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
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
    markersize        : float, optional, default: None
        The marker size in points**2 of the scatter plot.

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

    # Estimate posterior means and stds
    post_means = estimates.mean(axis=1)
    post_vars = estimates.var(axis=1, ddof=1)
    post_stds = np.sqrt(post_vars)

    # Estimate prior variance
    prior_vars = targets.var(axis=0, keepdims=True, ddof=1)

    # Compute contraction and z-score
    contraction = np.clip(1 - (post_vars / prior_vars), 0, 1)
    z_score = (post_means - targets) / post_stds

    # Loop and plot
    for i, ax in enumerate(plot_data["axes"].flat):
        if i >= plot_data["num_variables"]:
            break

        ax.scatter(contraction[:, i], z_score[:, i], color=color, alpha=0.5, s=markersize)
        ax.set_xlim([-0.05, 1.05])

    prettify_subplots(plot_data["axes"], num_subplots=plot_data["num_variables"], tick_fontsize=tick_fontsize)

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


class ZScoreContraction(CellPlot):
    """Class-based posterior z-score vs. contraction plot with cell/grid interface.

    For each variable, plots the posterior z-score against the posterior
    contraction as a graphical check for global model sensitivity, following [1].

    The posterior z-score measures how many posterior standard deviations the
    posterior mean is from the true value; the posterior contraction measures
    how much the posterior has narrowed relative to the prior.

    [1] Schad, D. J., Betancourt, M., & Vasishth, S. (2021). Toward a
    principled Bayesian workflow in cognitive science. Psychological
    methods, 26(1), 103. https://arxiv.org/abs/1904.12765

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
    color : str, optional (default = "#132a70")
        Color for the scatter points.
    markersize : float, optional
        Marker size in points**2.
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
        color: str = "#132a70",
        markersize: float = None,
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
            xlabel="Posterior contraction",
            ylabel="Posterior z-score",
        )
        self.color = color
        self.markersize = markersize

    def plot_cell(
        self,
        ax: plt.Axes,
        estimates_i: np.ndarray,
        targets_i: np.ndarray,
        variable_name: str = None,
        *,
        legend: bool = False,
    ) -> plt.Axes:
        """Draw the z-score vs. contraction scatter for a single variable.

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
        post_mean = estimates_i.mean(axis=1)
        post_var = estimates_i.var(axis=1, ddof=1)
        post_std = np.sqrt(post_var)

        prior_var = targets_i.var(ddof=1)

        contraction = np.clip(1 - (post_var / prior_var), 0, 1)
        z_score = (post_mean - targets_i) / post_std

        ax.scatter(contraction, z_score, color=self.color, alpha=0.5, s=self.markersize)
        ax.set_xlim([-0.05, 1.05])

        if variable_name is not None:
            ax.set_title(variable_name, fontsize=self.title_fontsize)

        return ax
