from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import kstest

from ...utils.dict_utils import dicts_to_arrays


def _l2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((a - b) ** 2, axis=-1))


def accuracy_random_points(
    estimates: Mapping[str, np.ndarray] | np.ndarray,
    targets: Mapping[str, np.ndarray] | np.ndarray,
    variable_keys: Sequence[str] = None,
    variable_names: Sequence[str] = None,
    references: np.ndarray = None,
    resolution: int = 20,
    standardize: bool = True,
    distance: Callable = None,
    rng: np.random.Generator = None,
) -> dict:
    """
    Computes the TARP (Test of Accuracy with Random Points) diagnostic.

    TARP is a global diagnostic which checks whether the estimated posteriors
    are calibrated by measuring expected coverage probabilities (ECP) across
    credibility levels [1].

    When no reference points are provided, they are generated as a derangement
    of the target parameters via a random cyclic shift (a pure permutation with
    no fixed points). For full control, e.g. to draw references from an independent
    prior sample, pass explicit ``references``.

    References:
    [1] Lemos et al., 2023, https://proceedings.mlr.press/v202/lemos23a/lemos23a.pdf

    Parameters
    ----------
    estimates  : np.ndarray or dict[str, np.ndarray]
        The random draws from the approximate posteriors, either as a NumPy array of
        shape (num_datasets, num_draws, num_variables) or as a dictionary mapping variable
        names to arrays, over ``num_datasets``.
    targets : np.ndarray or dict[str, np.ndarray]
        The corresponding ground-truth values sampled from the prior, either as a NumPy
        array of shape (num_datasets, num_variables) or as a dictionary mapping variable
        names to arrays.
    variable_keys : Sequence[str], optional (default = None)
       Select keys from the dictionaries provided in estimates and targets.
       By default, select all keys.
    variable_names : Sequence[str], optional (default = None)
        Optional variable names to show in the output.
    references : np.ndarray, optional (default = None)
        Reference points of shape ``(num_datasets, num_variables)``. If
        ``None``, reference points are generated automatically as a
        derangement of ``targets`` via a random cyclic shift (i.e. a pure
        permutation with no fixed points): a random offset in
        ``[1, num_datasets)`` is drawn and ``np.roll`` is applied along the
        dataset axis, so each dataset is assigned a reference from a
        *different* dataset.
    resolution    : int, optional, default: 20
        The number of credibility intervals (CIs) to consider
    standardize : bool, optional (default = True)
        Whether to normalize parameters to ``[0, 1]`` before computing
        distances.
    distance : callable, optional (default = L2)
        Distance function accepting two arrays of shape ``(..., D)`` and
        returning distances of shape ``(...,)``. Defaults to Euclidean (L2).
    rng : np.random.Generator, optional (default = None)

    Returns
    -------
    result : dict
        Dictionary containing:

        - "atc" : float
            Area to curve: signed difference between ECP and the diagonal
            for alpha > 0.5. Values near 0 indicate calibration;
            positive values indicate overdispersion; negative values indicate
            underdispersion.
        - "ks_pvalue" : float
            p-value of a two-sample Kolmogorov-Smirnov test between ECP and
            alpha. Values close to 1 indicate calibration.
        - "metric_name" : str
            The name of the metric ("TARP").
        - "variable_names" : str
            The (inferred) variable names.
    """
    if distance is None:
        distance = _l2

    samples = dicts_to_arrays(
        estimates=estimates,
        targets=targets,
        variable_keys=variable_keys,
        variable_names=variable_names,
    )

    thetas = samples["targets"]
    posterior_samples = samples["estimates"]
    num_datasets, num_draws, _ = posterior_samples.shape

    if references is None:
        rng = np.random.default_rng(rng)
        shift_id = rng.integers(1, num_datasets)
        references = np.roll(thetas, shift=shift_id, axis=0)

    if standardize:
        _mean = thetas.mean(axis=0, keepdims=True)
        _std = thetas.std(axis=0, keepdims=True)
        posterior_samples = (posterior_samples - _mean[:, None, :]) / _std[:, None, :]
        references = (references - _mean) / _std
        thetas = (thetas - _mean) / _std

    sample_dists = distance(references[:, None, :], posterior_samples)
    theta_dists = distance(references, thetas)

    # fraction of posterior samples closer to reference than true theta
    coverage_values = np.sum(sample_dists < theta_dists[:, None], axis=1) / num_draws

    hist, alpha_grid = np.histogram(coverage_values, bins=resolution, density=True)

    # empirical CDF via cumsum, prepend 0 to match alpha_grid length
    ecp = np.cumsum(hist) / hist.sum()
    ecp = np.concatenate([[0.0], ecp])

    midindex = alpha_grid.shape[0] // 2
    dalpha = alpha_grid[1] - alpha_grid[0]
    atc = np.sum((ecp[midindex:] - alpha_grid[midindex:]) * dalpha)

    ks_pvalue = kstest(ecp, alpha_grid)[1]

    variable_names = samples["estimates"].variable_names
    return {
        "values": atc,
        "ks_pvalue": ks_pvalue,
        "metric_name": "TARP",
        "variable_names": variable_names,
    }
