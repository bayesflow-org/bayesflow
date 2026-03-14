from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence

import numpy as np
import seaborn as sns

from ...utils.dict_utils import dicts_to_arrays


class PairPlot(ABC):
    """Abstract base class for BayesFlow pair-grid diagnostic plots.

    Returns ``seaborn.PairGrid`` objects rather than ``matplotlib.Figure``
    instances. The layout is controlled by ``height`` (the size of each facet)
    rather than a ``figsize`` / ``num_row`` / ``num_col`` grid, which is how
    seaborn PairGrids work.

    Unlike :class:`Plot`, test-quantity preprocessing is not included because
    pair plots operate directly on sample arrays and do not share that pattern.

    Parameters
    ----------
    variable_keys : Sequence[str], optional
        Select keys from the input dictionaries. By default, all keys.
    variable_names : Sequence[str], optional
        Human-readable variable names used in axis labels.
    height : float, optional (default = 2.5)
        Height in inches of each facet in the PairGrid.
    label_fontsize : int, optional (default = 14)
        Font size for axis labels.
    tick_fontsize : int, optional (default = 12)
        Font size for tick labels.
    """

    def __init__(
        self,
        variable_keys: Sequence[str] = None,
        variable_names: Sequence[str] = None,
        height: float = 2.5,
        label_fontsize: int = 14,
        tick_fontsize: int = 12,
    ):
        self.variable_keys = variable_keys
        self.variable_names = variable_names
        self.height = height
        self.label_fontsize = label_fontsize
        self.tick_fontsize = tick_fontsize

    def _preprocess(
        self,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        targets: Mapping[str, np.ndarray] | np.ndarray = None,
        priors: Mapping[str, np.ndarray] | np.ndarray = None,
        dataset_id: int = None,
    ) -> dict:
        """Convert dicts/arrays to a standardised samples dict.

        Parameters
        ----------
        estimates : array-like
            Posterior draws or sample array.
        targets : array-like, optional
            Ground-truth values.
        priors : array-like, optional
            Prior samples (used by :class:`PairsPosterior`).
        dataset_id : int, optional
            Select a single dataset by index.

        Returns
        -------
        dict as returned by :func:`~bayesflow.utils.dict_utils.dicts_to_arrays`.
        """
        return dicts_to_arrays(
            estimates=estimates,
            targets=targets,
            priors=priors,
            dataset_ids=dataset_id,
            variable_keys=self.variable_keys,
            variable_names=self.variable_names,
        )

    @abstractmethod
    def create(
        self,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        targets: Mapping[str, np.ndarray] | np.ndarray = None,
    ) -> sns.PairGrid:
        """Create and return the pair-grid figure.

        Parameters
        ----------
        estimates : array-like
            Posterior draws or sample array.
        targets : array-like, optional
            Ground-truth values.

        Returns
        -------
        seaborn.PairGrid
        """
        raise NotImplementedError

    def __call__(
        self,
        estimates: Mapping[str, np.ndarray] | np.ndarray,
        targets: Mapping[str, np.ndarray] | np.ndarray = None,
    ) -> sns.PairGrid:
        return self.create(estimates, targets)

    def save(
        self,
        grid: sns.PairGrid,
        path: str,
        **kwargs,
    ) -> sns.PairGrid:
        """Save an already-created PairGrid to ``path``.

        Because :meth:`create` signatures vary across subclasses, this method
        accepts the grid directly rather than recreating it.

        Parameters
        ----------
        grid : seaborn.PairGrid
            The grid returned by :meth:`create`.
        path : str
            Destination file path (format inferred from extension).
        **kwargs
            Additional keyword arguments forwarded to ``grid.savefig``.

        Returns
        -------
        seaborn.PairGrid
            The same grid, for convenience.

        Examples
        --------
        >>> grid = my_pair_plot.create(estimates, targets)
        >>> my_pair_plot.save(grid, "pairs.pdf", dpi=150)
        """
        grid.savefig(path, **kwargs)
        return grid
