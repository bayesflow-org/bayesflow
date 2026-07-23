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
    figsize: tuple = None,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    value_fontsize: int = 10,
    tick_fontsize: int = 12,
    cmap: matplotlib.colors.Colormap | str = None,
    fmt: str = ".1f",
    title: bool = True,
    ax: plt.Axes = None,
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
    figsize : tuple or None, optional
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
        name of a registered colormap.  Default (``None``) uses a diverging
        gray-white-blue colormap; its blue end is the BayesFlow navy for a
        standalone plot, switching to the brighter Bayes factor accent color
        (``#6969ff``) when drawn into a provided ``ax`` (e.g. side by side with
        the confusion matrix).  Blue indicates evidence in favour of the true model.
    fmt : str, optional
        Format string for cell annotations (default: ``".1f"``).
    title : bool, optional
        Whether to add the plot title (default: True).
    ax : matplotlib.axes.Axes, optional
        An existing axis to draw into. If ``None``, a new figure and axis are
        created. When provided, ``figsize`` is ignored and the parent figure is
        returned, enabling composition (e.g. side-by-side panels).

    Returns
    -------
    fig : plt.Figure
    """
    if cmap is None:
        # Navy by default; switch to the brighter Bayes factor accent blue (#6969ff) only when
        # embedded into a caller-provided axis (e.g. side by side with the navy confusion matrix),
        # so a standalone plot stays consistent with the other (navy) diagnostics.
        blue = "#6969ff" if ax is not None else "#132a70"
        cmap = LinearSegmentedColormap.from_list("", ["dimgray", "white", blue])

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

    owns_figure = ax is None
    if owns_figure:
        if figsize is None:
            size = max(4.0, M * 1.2)
            figsize = (size + 0.5, size)
        fig, axes = make_figure(1, 1, figsize=figsize)
        ax = axes[0]
    else:
        fig = ax.figure

    im = ax.imshow(mean_matrix, interpolation="nearest", cmap=cmap, norm=norm)
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.75)
    cbar.ax.tick_params(labelsize=value_fontsize)

    ax.set_xticks(range(M))
    ax.set_xticklabels(model_names, fontsize=tick_fontsize)
    ax.set_yticks(range(M))
    ax.set_yticklabels(model_names, fontsize=tick_fontsize)
    ax.set_xlabel(r"Competing model", fontsize=label_fontsize)
    ax.set_ylabel(r"True model", fontsize=label_fontsize)

    for i in range(M):
        for j in range(M):
            val = mean_matrix[i, j]
            text_color = "white" if abs(val) > 0.6 * abs_max else "black"
            ax.text(j, i, format(val, fmt), ha="center", va="center", fontsize=value_fontsize, color=text_color)

    if title:
        ax.set_title(r"Mean log Bayes factor", fontsize=title_fontsize)

    # Only manage layout when we own the figure; otherwise leave it to the caller
    # so embedding into a shared figure does not fight the parent layout.
    if owns_figure:
        fig.tight_layout()
    return fig
