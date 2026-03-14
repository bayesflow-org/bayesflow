from collections.abc import Callable, Mapping, Sequence

import numpy as np
import matplotlib.pyplot as plt

from .plot import Plot


class CustomPlot(Plot):
    """Wraps a user-defined function as a :class:`Plot`.

    Provides automatic preprocessing (test-quantity computation and
    dict-to-array conversion) via the :class:`Plot` base class, so the
    wrapped function receives clean numpy arrays and can focus solely on
    the plotting logic.

    Parameters
    ----------
    fn : callable
        A function with signature::

            fn(estimates, targets, **kwargs) -> matplotlib.figure.Figure

        where ``estimates`` is a LabeledArray of shape
        ``(num_datasets, num_draws, num_variables)`` and ``targets`` is an
        array of shape ``(num_datasets, num_variables)``.
        ``variable_names`` can be retrieved from ``estimates.variable_names``.
    variable_keys : Sequence[str], optional
        Select keys from the input dictionaries. By default, all keys.
    variable_names : Sequence[str], optional
        Human-readable variable names for axis labels.
    test_quantities : dict[str, Callable], optional
        Functions to compute test quantities before the main computation.
    figsize : Sequence[float], optional
        Overall figure size.
    num_row : int, optional
        Number of subplot rows.
    num_col : int, optional
        Number of subplot columns.
    label_fontsize : int, optional (default = 16)
    title_fontsize : int, optional (default = 18)
    tick_fontsize : int, optional (default = 12)
    **kwargs
        Additional keyword arguments forwarded to ``fn`` at every call.

    Examples
    --------
    >>> def my_plot(estimates, targets):
    ...     fig, ax = plt.subplots()
    ...     ax.scatter(targets[:, 0], estimates.mean(axis=1)[:, 0])
    ...     return fig
    >>> plot = CustomPlot(my_plot)
    >>> fig = plot(estimates, targets)
    """

    def __init__(
        self,
        fn: Callable,
        variable_keys: Sequence[str] = None,
        variable_names: Sequence[str] = None,
        test_quantities: dict[str, Callable] = None,
        figsize: Sequence[float] = None,
        num_row: int = None,
        num_col: int = None,
        label_fontsize: int = 16,
        title_fontsize: int = 18,
        tick_fontsize: int = 12,
        **kwargs,
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
        self.fn = fn
        self.kwargs = kwargs

    def create(
        self,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        targets: Mapping[str, np.ndarray] | np.ndarray,
    ) -> plt.Figure:
        samples = self._preprocess(estimates, targets)
        return self.fn(samples["estimates"], samples["targets"], **self.kwargs)
