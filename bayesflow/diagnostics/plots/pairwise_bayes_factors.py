from collections.abc import Sequence

import matplotlib.colors
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from bayesflow.utils.plot_utils import make_figure


def pairwise_bayes_factors(
    pred_log_bayes_factors: np.ndarray,
    true_models: np.ndarray,
    model_names: Sequence[str] = None,
    fig_size: tuple = None,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    value_fontsize: int = 10,
    tick_fontsize: int = 12,
    cmap: matplotlib.colors.Colormap | str = None,
    fmt: str = ".1f",
    title: bool = True,
) -> plt.Figure:
    r"""Mean pairwise log Bayes factor heatmap, stratified by true generating model.

    For each true model :math:`\mathcal{M}_m`, computes the mean predicted
    log Bayes factor :math:`\log \mathrm{BF}_{m,j}` over all datasets generated from
    :math:`\mathcal{M}_m`:

    .. math::

        \hat{\mu}_{m,j} = \mathbb{E}\!\left[\log \mathrm{BF}_{m,j}(x) \mid \mathcal{M}_m\right]

    where :math:`\log \mathrm{BF}_{m,j} = f_m - f_j` and the full :math:`M \times M`
    pairwise matrix is obtained by prepending the reference anchor
    :math:`f_0 \equiv 0` to the :math:`M - 1` network outputs.

    A well-trained network produces **positive** off-diagonal entries in each
    row (the predicted evidence favours the true model over alternatives) and
    zeros on the diagonal (trivially, :math:`\mathrm{BF}_{m,m} = 1`).

    Parameters
    ----------
    pred_log_bayes_factors : np.ndarray of shape (num_datasets, num_models - 1)
        Predicted log Bayes factors :math:`\log \mathrm{BF}_{k,0}` for
        :math:`k = 1, \ldots, M-1`, as returned by
        :meth:`~bayesflow.workflows.ModelComparisonWorkflow.predict` with
        ``probs=False`` when a Bayes factor scoring rule is active.
    true_models : np.ndarray of shape (num_datasets, num_models) or (num_datasets,)
        One-hot encoded model labels or integer class indices.
    model_names : Sequence[str] or None, optional
        Human-readable model names. Defaults to :math:`M_1, M_2, \ldots`.
    fig_size : tuple or None, optional
        Figure size passed to matplotlib. Inferred from ``num_models`` if None.
    label_fontsize : int, optional
        Font size for axis labels (default: 16).
    title_fontsize : int, optional
        Font size for the plot title (default: 18).
    value_fontsize : int, optional
        Font size for the cell value annotations (default: 10).
    tick_fontsize : int, optional
        Font size for tick labels (default: 12).
    cmap : matplotlib.colors.Colormap or str, optional
        Colormap for the heatmap, always centred at zero via
        :class:`~matplotlib.colors.TwoSlopeNorm`.  If a str, it should be the
        name of a registered colormap.  Default (``None``) uses the BayesFlow
        white-to-blue colormap, matching :func:`mc_confusion_matrix`.
    fmt : str, optional
        Format string for cell annotations (default: ``".1f"``).
    title : bool, optional
        Whether to add the plot title (default: True).

    Returns
    -------
    fig : plt.Figure
    """
    if cmap is None:
        cmap = LinearSegmentedColormap.from_list("", ["#c0392b", "white", "#27ae60"])

    pred_log_bayes_factors = np.asarray(pred_log_bayes_factors)
    true_models = np.asarray(true_models)

    if true_models.ndim == 2:
        true_model_idx = np.argmax(true_models, axis=-1)
    else:
        true_model_idx = true_models.astype(int)

    N, M_minus_1 = pred_log_bayes_factors.shape
    M = M_minus_1 + 1

    if model_names is None:
        model_names = [rf"$M_{{{m}}}$" for m in range(1, M + 1)]

    # Prepend f_0 = 0 to obtain (N, M) with entry k = log BF_{k,0}
    f0 = np.zeros((N, 1), dtype=pred_log_bayes_factors.dtype)
    f = np.concatenate([f0, pred_log_bayes_factors], axis=-1)

    # Full (N, M, M) pairwise matrix: entry [n, i, j] = f_i - f_j = log BF_{i,j}
    pairwise = f[:, :, np.newaxis] - f[:, np.newaxis, :]

    # Stratified mean: row m = mean log BF_{m,j} over datasets from M_m
    mean_matrix = np.zeros((M, M))
    for m in range(M):
        mask = true_model_idx == m
        if mask.sum() > 0:
            mean_matrix[m] = pairwise[mask, m, :].mean(axis=0)

    # Diverging normalisation centred at zero
    abs_max = np.abs(mean_matrix).max()
    if abs_max == 0.0:
        abs_max = 1.0
    norm = matplotlib.colors.TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)

    if fig_size is None:
        size = max(4.0, M * 1.2)
        fig_size = (size + 0.5, size)

    fig, axes = make_figure(1, 1, figsize=fig_size)
    ax = axes[0]

    im = ax.imshow(mean_matrix, interpolation="nearest", cmap=cmap, norm=norm)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75)
    cbar.ax.tick_params(labelsize=value_fontsize)

    ax.set_xticks(range(M))
    ax.set_xticklabels(model_names, fontsize=tick_fontsize)
    ax.set_yticks(range(M))
    ax.set_yticklabels(model_names, fontsize=tick_fontsize)
    ax.set_xlabel(r"Comparison model $\mathcal{M}_j$", fontsize=label_fontsize)
    ax.set_ylabel(r"True model $\mathcal{M}_m$", fontsize=label_fontsize)

    for i in range(M):
        for j in range(M):
            val = mean_matrix[i, j]
            text_color = "white" if abs(val) > 0.6 * abs_max else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=value_fontsize, color=text_color)

    if title:
        ax.set_title(r"Mean log Bayes factor by true model", fontsize=title_fontsize)

    fig.tight_layout()
    return fig
