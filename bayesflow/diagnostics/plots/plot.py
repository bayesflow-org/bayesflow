from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt

from ...utils.dict_utils import dicts_to_arrays, compute_test_quantities
from ...utils.plot_utils import set_layout, make_figure, prettify_subplots, add_titles_and_labels


class Plot(ABC):
    """Abstract base class for BayesFlow diagnostic plots.

    Subclasses implement :meth:`create` and receive preprocessed arrays via
    :meth:`_preprocess`, which handles the shared boilerplate of test-quantity
    computation and dict-to-array conversion.

    Parameters
    ----------
    variable_keys : Sequence[str], optional
        Select keys from the dictionaries provided in ``estimates`` and
        ``targets``. By default, all keys are selected.
    variable_names : Sequence[str], optional
        Human-readable variable names used in axis labels and titles.
    test_quantities : dict[str, Callable], optional
        A dict mapping titles to functions that compute test quantities from
        estimate/target draws. Keys are appended to ``variable_keys`` and
        ``variable_names`` automatically.
    figsize : Sequence[float], optional
        Overall figure size passed to ``plt.subplots``.
    num_row : int, optional
        Number of rows in the subplot grid. Inferred automatically if not set.
    num_col : int, optional
        Number of columns in the subplot grid. Inferred automatically if not set.
    label_fontsize : int, optional (default = 16)
        Font size for axis labels.
    title_fontsize : int, optional (default = 18)
        Font size for subplot titles.
    tick_fontsize : int, optional (default = 12)
        Font size for tick labels.
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
    ):
        self.variable_keys = variable_keys
        self.variable_names = variable_names
        self.test_quantities = test_quantities
        self.figsize = figsize
        self.num_row = num_row
        self.num_col = num_col
        self.label_fontsize = label_fontsize
        self.title_fontsize = title_fontsize
        self.tick_fontsize = tick_fontsize

    def _preprocess(
        self,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        targets: Mapping[str, np.ndarray] | np.ndarray,
    ) -> dict:
        """Run test-quantity computation and dict-to-array conversion.

        Returns
        -------
        dict with keys ``"estimates"`` (LabeledArray of shape
        ``(num_datasets, num_draws, num_variables)``) and ``"targets"``
        (array of shape ``(num_datasets, num_variables)``).
        """
        variable_keys = self.variable_keys
        variable_names = self.variable_names

        if self.test_quantities is not None:
            updated = compute_test_quantities(
                targets=targets,
                estimates=estimates,
                variable_keys=variable_keys,
                variable_names=variable_names,
                test_quantities=self.test_quantities,
            )
            variable_keys = updated["variable_keys"]
            variable_names = updated["variable_names"]
            estimates = updated["estimates"]
            targets = updated["targets"]

        return dicts_to_arrays(
            estimates=estimates,
            targets=targets,
            variable_keys=variable_keys,
            variable_names=variable_names,
        )

    @abstractmethod
    def create(
        self,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        targets: Mapping[str, np.ndarray] | np.ndarray,
    ) -> plt.Figure:
        """Create and return the diagnostic figure.

        Parameters
        ----------
        estimates : array-like of shape (num_datasets, num_draws, num_variables)
            Posterior draws.
        targets : array-like of shape (num_datasets, num_variables)
            Ground-truth values.

        Returns
        -------
        matplotlib.figure.Figure
        """
        raise NotImplementedError

    def __call__(
        self,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        targets: Mapping[str, np.ndarray] | np.ndarray,
    ) -> plt.Figure:
        return self.create(estimates, targets)

    def save(
        self,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        targets: Mapping[str, np.ndarray] | np.ndarray,
        path: str,
        **kwargs,
    ) -> plt.Figure:
        """Create the figure and save it to ``path``.

        Parameters
        ----------
        estimates : array-like
            Posterior draws, forwarded to :meth:`create`.
        targets : array-like
            Ground-truth values, forwarded to :meth:`create`.
        path : str
            Destination file path (format inferred from extension).
        **kwargs
            Additional keyword arguments forwarded to ``fig.savefig``.

        Returns
        -------
        matplotlib.figure.Figure
            The created figure.
        """
        fig = self.create(estimates, targets)
        fig.savefig(path, **kwargs)
        return fig


class CellPlot(Plot):
    """Grid-level plot that delegates per-variable drawing to :meth:`plot_cell`.

    Provides a default :meth:`create` implementation that handles preprocessing,
    subplot-grid creation, iteration over variables, and layout finalization.
    Subclasses implement :meth:`plot_cell` to draw the diagnostic for a single
    variable on a given :class:`matplotlib.axes.Axes`.

    This follows the Seaborn convention of separating axes-level (cell) from
    figure-level (grid) logic: :meth:`plot_cell` is the axes-level function and
    :meth:`create` is the figure-level function.

    Parameters
    ----------
    variable_keys : Sequence[str], optional
        Select keys from the dictionaries provided in ``estimates`` and
        ``targets``. By default, all keys are selected.
    variable_names : Sequence[str], optional
        Human-readable variable names used in subplot titles.
    test_quantities : dict[str, Callable], optional
        A dict mapping titles to functions that compute test quantities from
        estimate/target draws.
    figsize : Sequence[float], optional
        Overall figure size passed to ``plt.subplots``.
    num_row : int, optional
        Number of rows in the subplot grid. Inferred automatically if not set.
    num_col : int, optional
        Number of columns in the subplot grid. Inferred automatically if not set.
    label_fontsize : int, optional (default = 16)
        Font size for axis labels.
    title_fontsize : int, optional (default = 18)
        Font size for subplot titles.
    tick_fontsize : int, optional (default = 12)
        Font size for tick labels.
    xlabel : str, optional
        Label for the x-axis (applied to bottom-row subplots).
    ylabel : str, optional
        Label for the y-axis (applied to left-column subplots).
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
        xlabel: str = None,
        ylabel: str = None,
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
        )
        self.xlabel = xlabel
        self.ylabel = ylabel

    @abstractmethod
    def plot_cell(
        self,
        ax: plt.Axes,
        estimates_i: np.ndarray,
        targets_i: np.ndarray,
        variable_name: str = None,
        *,
        legend: bool = False,
    ) -> plt.Axes:
        """Draw the diagnostic for a single variable on the given axes.

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
            Whether to add a legend to this cell. Typically ``True`` for the
            first cell only, as set by :meth:`create`.

        Returns
        -------
        matplotlib.axes.Axes
        """
        raise NotImplementedError

    def create(
        self,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        targets: Mapping[str, np.ndarray] | np.ndarray,
    ) -> plt.Figure:
        """Create and return the diagnostic figure.

        Preprocesses inputs, builds the subplot grid, calls :meth:`plot_cell`
        for each variable, then applies shared layout formatting.

        Parameters
        ----------
        estimates : dict or array of shape (num_datasets, num_draws, num_variables)
            Posterior draws.
        targets : dict or array of shape (num_datasets, num_variables)
            Ground-truth values.

        Returns
        -------
        matplotlib.figure.Figure
        """
        data = self._preprocess(estimates, targets)
        estimates_arr = data["estimates"]
        targets_arr = data["targets"]
        variable_names = list(estimates_arr.variable_names)
        num_variables = len(variable_names)

        num_row, num_col = set_layout(num_variables, self.num_row, self.num_col)
        fig, axes = make_figure(num_row, num_col, figsize=self.figsize)
        axes = np.atleast_1d(axes)

        for i, ax in enumerate(axes.flat):
            if i >= num_variables:
                break
            self.plot_cell(
                ax,
                np.asarray(estimates_arr[:, :, i]),
                np.asarray(targets_arr[:, i]),
                variable_name=variable_names[i],
                legend=(i == 0),
            )

        prettify_subplots(axes, num_subplots=num_variables, tick_fontsize=self.tick_fontsize)
        add_titles_and_labels(
            axes,
            num_row,
            num_col,
            xlabel=self.xlabel,
            ylabel=self.ylabel,
            label_fontsize=self.label_fontsize,
        )
        fig.tight_layout()
        return fig
