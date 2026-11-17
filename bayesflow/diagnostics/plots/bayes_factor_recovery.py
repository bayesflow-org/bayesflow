from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from bayesflow.utils.plot_utils import make_figure, make_quadratic, add_metric, prettify_subplots, add_titles_and_labels


def bayes_factor_recovery(
    pred_log_bayes_factors: np.ndarray,
    true_log_bayes_factors: np.ndarray,
    true_models: np.ndarray,
    model_names: Sequence[str] = None,
    reference_model: int = 0,
    add_corr: bool = True,
    figsize: tuple = None,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    metric_fontsize: int = 14,
    tick_fontsize: int = 12,
    alpha: float = 0.5,
    markersize: float = None,
    num_col: int = None,
    num_row: int = None,
) -> plt.Figure:
    """Scatter plot of true vs. predicted log Bayes factors for M-model comparison.

    Produces one subplot per competing model, each showing the predicted log Bayes
    factor against the analytically or numerically exact value.  Points are coloured
    by the true generating model.  Quadrant shading distinguishes correct decisions
    (true and predicted Bayes factors favour the same model) from incorrect ones.

    Parameters
    ----------
    pred_log_bayes_factors : np.ndarray of shape (num_datasets, num_models - 1)
        Predicted log Bayes factors ``log BF_{k, reference}`` for
        ``k != reference``, as output by a Bayes factor scoring rule network.
    true_log_bayes_factors : np.ndarray of shape (num_datasets, num_models - 1)
        Ground-truth log Bayes factors with the same column ordering as
        ``pred_log_bayes_factors``.
    true_models : np.ndarray of shape (num_datasets, num_models) or (num_datasets,)
        Either one-hot encoded model labels or integer class indices.
    model_names : Sequence[str] or None, optional
        Human-readable names for all ``num_models`` models, used for axis labels
        and the legend.  Defaults to ``M_1, M_2, ...``.
    reference_model : int, optional
        Index (0-based) of the reference model against which all log Bayes factors
        are computed (default: 0).
    add_corr : bool, optional
        Annotate each panel with the Pearson correlation between true and
        predicted log Bayes factors (default: True).
    figsize : tuple or None, optional
        Passed to ``matplotlib``.  Inferred from the number of panels if None.
    label_fontsize : int, optional
        Font size for axis labels (default: 16).
    title_fontsize : int, optional
        Font size for subplot titles (default: 18).
    metric_fontsize : int, optional
        Font size for the correlation annotation (default: 14).
    tick_fontsize : int, optional
        Font size for tick labels (default: 12).
    alpha : float, optional
        Scatter point opacity (default: 0.5).
    markersize : float or None, optional
        Scatter point size in points (default: None → matplotlib default).
    num_col : int or None, optional
        Number of subplot columns.  Inferred if None.
    num_row : int or None, optional
        Number of subplot rows.  Inferred if None.

    Returns
    -------
    fig : plt.Figure
    """
    pred_log_bayes_factors = np.asarray(pred_log_bayes_factors)
    true_log_bayes_factors = np.asarray(true_log_bayes_factors)
    true_models = np.asarray(true_models)

    if pred_log_bayes_factors.shape != true_log_bayes_factors.shape:
        raise ValueError(
            f"pred_log_bayes_factors and true_log_bayes_factors must have the same shape, "
            f"got {pred_log_bayes_factors.shape} and {true_log_bayes_factors.shape}."
        )

    num_panels = pred_log_bayes_factors.shape[1]
    num_models = num_panels + 1

    # Accept one-hot or integer labels
    if true_models.ndim == 2:
        true_model_idx = np.argmax(true_models, axis=-1)
    else:
        true_model_idx = true_models.astype(int)

    if model_names is None:
        model_names = [rf"$M_{{{m}}}$" for m in range(1, num_models + 1)]

    # Build subplot titles, e.g. "M_2 vs M_1"
    ref_name = model_names[reference_model]
    competing = [model_names[k] for k in range(num_models) if k != reference_model]
    panel_titles = [f"{name} vs. {ref_name}" for name in competing]

    # Categorical colors — one per model
    cmap = plt.colormaps["tab10"].resampled(num_models)
    model_colors = [cmap(m) for m in range(num_models)]

    # Layout
    if num_col is None and num_row is None:
        num_col = min(num_panels, 3)
        num_row = int(np.ceil(num_panels / num_col))
    elif num_col is None:
        num_col = int(np.ceil(num_panels / num_row))
    elif num_row is None:
        num_row = int(np.ceil(num_panels / num_col))

    fig, axes = make_figure(num_row, num_col, figsize=figsize)
    axes_flat = np.asarray(axes).flat

    scatter_kw = dict(alpha=alpha)
    if markersize is not None:
        scatter_kw["s"] = markersize**2

    for panel_idx, ax in enumerate(axes_flat):
        if panel_idx >= num_panels:
            break

        x = true_log_bayes_factors[:, panel_idx]
        y = pred_log_bayes_factors[:, panel_idx]

        # Per-model scatter
        for m in range(num_models):
            mask = true_model_idx == m
            if not mask.any():
                continue
            ax.scatter(
                x[mask],
                y[mask],
                color=model_colors[m],
                label=model_names[m],
                **scatter_kw,
            )

        # Make axes square and add identity line
        make_quadratic(ax, x, y)
        xlim, ylim = ax.get_xlim(), ax.get_ylim()

        # Quadrant shading: correct = same sign (Q1 and Q3), incorrect = opposite.
        # Bayes factor accent blue (#6969ff), distinct from the navy used elsewhere.
        ax.fill_between(
            [0, xlim[1]],
            0,
            ylim[1],
            color="#6969ff",
            alpha=0.10,
            zorder=0,
        )
        ax.fill_between(
            [xlim[0], 0],
            ylim[0],
            0,
            color="#6969ff",
            alpha=0.10,
            zorder=0,
        )
        ax.fill_between(
            [xlim[0], 0],
            0,
            ylim[1],
            color="gray",
            alpha=0.15,
            zorder=0,
        )
        ax.fill_between(
            [0, xlim[1]],
            ylim[0],
            0,
            color="gray",
            alpha=0.15,
            zorder=0,
        )

        ax.axhline(0, color="gray", linewidth=0.8, linestyle=":", zorder=1)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle=":", zorder=1)

        if add_corr:
            corr = np.corrcoef(x, y)[0, 1]
            add_metric(ax, metric_text="$r$", metric_value=corr, metric_fontsize=metric_fontsize)

        ax.set_title(panel_titles[panel_idx], fontsize=title_fontsize)

    prettify_subplots(np.asarray(axes), num_subplots=num_panels, tick_fontsize=tick_fontsize)
    add_titles_and_labels(
        axes=np.asarray(axes),
        num_row=num_row,
        num_col=num_col,
        xlabel=f"True $\\log \\mathrm{{BF}}_{{k,{reference_model}}}$",
        ylabel=f"Predicted $\\log \\mathrm{{BF}}_{{k,{reference_model}}}$",
        label_fontsize=label_fontsize,
    )

    # Single shared legend below the figure
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=model_colors[m], markersize=8, label=model_names[m])
        for m in range(num_models)
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=num_models,
        fontsize=tick_fontsize,
        frameon=False,
    )

    fig.tight_layout()
    return fig
