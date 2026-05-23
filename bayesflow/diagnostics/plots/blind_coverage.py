from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np

from bayesflow.utils.ecdf import simultaneous_ecdf_bands
from bayesflow.utils.plot_utils import make_figure, prettify_subplots, add_titles_and_labels


def blind_coverage(
    pred_log_bayes_factors: np.ndarray,
    true_models: np.ndarray,
    model_names: Sequence[str] = None,
    reference_model: int = 0,
    num_quantile_points: int = 200,
    confidence: float = 0.95,
    difference: bool = True,
    figsize: tuple = None,
    label_fontsize: int = 16,
    title_fontsize: int = 18,
    legend_fontsize: int = 12,
    tick_fontsize: int = 12,
    fill_color: str = "grey",
    num_col: int = None,
    num_row: int = None,
) -> plt.Figure:
    r"""Blind coverage test for amortized Bayes factor estimators.

    Introduced in Jeffrey & Wandelt (2024) as a gold-standard diagnostic for
    evidence networks.  For each pairwise log Bayes factor
    :math:`\log K_{k, \text{ref}}`, the test checks whether the network's
    predicted values correctly rank datasets by their true generating model —
    using **only** model identity labels, without requiring analytic
    ground-truth log Bayes factors.

    **What is plotted**

    For each pairwise comparison (one subplot per competing model :math:`k`):

    - *x-axis*: marginal quantile level :math:`\alpha \in [0, 1]`, computed
      from the full predicted log :math:`K` distribution without conditioning
      on the true model ("blind" thresholds).
    - *y-axis* for model :math:`m`:
      :math:`\hat{F}_m\!\left(\hat{F}^{-1}(\alpha)\right)` — the fraction of
      model-:math:`m` datasets whose predicted log :math:`K` falls at or below
      the :math:`\alpha`-quantile of the *marginal* distribution.
    - *Grey band*: simultaneous confidence band around the diagonal under the
      null hypothesis of no discrimination (all model groups share the same
      predicted log :math:`K` distribution).

    **Interpretation**

    +------------------------------+-----------------------------------+
    | Model group                  | Expected curve (good estimator)   |
    +==============================+===================================+
    | Reference model :math:`M_0` | *Above* the diagonal              |
    | (should rank low)            |                                   |
    +------------------------------+-----------------------------------+
    | Competing model :math:`M_k` | *Below* the diagonal              |
    | (should rank high)           |                                   |
    +------------------------------+-----------------------------------+
    | Other models                 | Between the two extremes          |
    +------------------------------+-----------------------------------+

    All curves lying on the diagonal means the estimator provides no
    discrimination (equivalent to a random classifier).

    Parameters
    ----------
    pred_log_bayes_factors : np.ndarray of shape (num_datasets, num_models - 1)
        Predicted log Bayes factors :math:`\log K_{k, \text{ref}}` for each
        competing model :math:`k`, as returned by
        :meth:`~bayesflow.workflows.ModelComparisonWorkflow.predict` when a
        Bayes factor scoring rule is active.
    true_models : np.ndarray of shape (num_datasets, num_models) or (num_datasets,)
        One-hot encoded model labels or integer class indices.
    model_names : Sequence[str] or None, optional
        Human-readable names for all ``num_models`` models.
        Defaults to :math:`M_1, M_2, \ldots`.
    reference_model : int, optional
        Index (0-based) of the reference model against which log Bayes factors
        are computed (default: 0).
    num_quantile_points : int, optional
        Number of :math:`\alpha` values at which to evaluate the conditional
        ECDFs (default: 200).
    confidence : float, optional
        Confidence level for the simultaneous bands under the null
        (default: 0.95).
    difference : bool, optional
        If ``True`` (default), plot the deviation from the diagonal
        (:math:`\hat{F}_m(\hat{F}^{-1}(\alpha)) - \alpha`) so that the
        reference line is a flat zero and departures are immediately visible.
        If ``False``, plot the raw conditional ECDF on the unit square.
    figsize : tuple or None, optional
        Passed to ``matplotlib``. Inferred from the number of panels if None.
    label_fontsize : int, optional
        Font size for axis labels (default: 16).
    title_fontsize : int, optional
        Font size for subplot titles (default: 18).
    legend_fontsize : int, optional
        Font size for the legend (default: 12).
    tick_fontsize : int, optional
        Font size for tick labels (default: 12).
    fill_color : str, optional
        Colour of the simultaneous confidence band (default: ``"grey"``).
    num_col : int or None, optional
        Number of subplot columns. Inferred if None.
    num_row : int or None, optional
        Number of subplot rows. Inferred if None.

    Returns
    -------
    fig : plt.Figure

    References
    ----------
    Jeffrey, N. & Wandelt, B. D. (2024). Evidence Networks: Handling Nuisance
    Parameters for Likelihood-Free Hypothesis Tests.
    """
    pred_log_bayes_factors = np.asarray(pred_log_bayes_factors)
    true_models = np.asarray(true_models)

    if true_models.ndim == 2:
        true_model_idx = np.argmax(true_models, axis=-1)
    else:
        true_model_idx = true_models.astype(int)

    num_datasets, num_panels = pred_log_bayes_factors.shape
    num_models = num_panels + 1

    if model_names is None:
        model_names = [rf"$M_{{{m}}}$" for m in range(1, num_models + 1)]

    ref_name = model_names[reference_model]
    competing = [model_names[k] for k in range(num_models) if k != reference_model]
    panel_titles = [f"{name} vs. {ref_name}" for name in competing]

    # Layout
    if num_col is None and num_row is None:
        num_col = min(num_panels, 3)
        num_row = int(np.ceil(num_panels / num_col))
    elif num_col is None:
        num_col = int(np.ceil(num_panels / num_row))
    elif num_row is None:
        num_row = int(np.ceil(num_panels / num_col))

    fig, axes = make_figure(num_row, num_col, figsize=figsize)

    # Quantile levels to evaluate
    alphas = np.linspace(0.0, 1.0, num_quantile_points)

    # Categorical colour map — one colour per model (matches bayes_factor_recovery)
    cmap = plt.cm.get_cmap("tab10", num_models)
    model_colors = [cmap(m) for m in range(num_models)]

    for panel_idx, ax in enumerate(np.asarray(axes).flat):
        if panel_idx >= num_panels:
            break

        log_k = pred_log_bayes_factors[:, panel_idx]

        # Blind quantile thresholds from the full marginal distribution
        thresholds = np.quantile(log_k, alphas)

        # Simultaneous bands under the null (no discrimination):
        # use the smallest per-model group size for a conservative bound
        group_sizes = [int(np.sum(true_model_idx == m)) for m in range(num_models)]
        min_group_size = max(min(group_sizes), 2)
        _, z_band, lower_band, upper_band = simultaneous_ecdf_bands(
            num_estimates=min_group_size,
            confidence=confidence,
        )

        if difference:
            band_lo = lower_band - z_band
            band_hi = upper_band - z_band
        else:
            band_lo = lower_band
            band_hi = upper_band

        ax.fill_between(
            z_band,
            band_lo,
            band_hi,
            color=fill_color,
            alpha=0.25,
            label=rf"{int(confidence * 100)}\% simultaneous band",
            zorder=0,
        )
        if difference:
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
        else:
            ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)

        panel_y_vals = []
        for m in range(num_models):
            mask_m = true_model_idx == m
            log_k_m = log_k[mask_m]
            if len(log_k_m) == 0:
                continue

            # Conditional ECDF evaluated at each blind threshold
            cond_ecdf = np.array([np.mean(log_k_m <= t) for t in thresholds])
            y_vals = cond_ecdf - alphas if difference else cond_ecdf
            if difference:
                panel_y_vals.append(y_vals)

            ax.plot(
                alphas,
                y_vals,
                color=model_colors[m],
                label=model_names[m],
                linewidth=1.8,
                zorder=2,
            )

        ax.set_xlim(0.0, 1.0)
        if difference and panel_y_vals:
            all_y = np.concatenate(panel_y_vals)
            ax.set_ylim(np.nanmin(all_y), np.nanmax(all_y))
        else:
            ax.set_ylim(0.0, 1.0)
        ax.set_title(panel_titles[panel_idx], fontsize=title_fontsize)

    prettify_subplots(np.asarray(axes), num_subplots=num_panels, tick_fontsize=tick_fontsize)
    if difference:
        ylabel = r"$\hat{F}_m(\hat{F}^{-1}(\alpha)) - \alpha$"
    else:
        ylabel = r"Conditional ECDF $\hat{F}_m(\hat{F}^{-1}(\alpha))$"
    add_titles_and_labels(
        axes=np.asarray(axes),
        num_row=num_row,
        num_col=num_col,
        xlabel=r"Marginal quantile $\alpha$ (blind threshold)",
        ylabel=ylabel,
        label_fontsize=label_fontsize,
    )

    handles, labels = np.asarray(axes).flat[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(handles),
        fontsize=legend_fontsize,
        bbox_to_anchor=(0.5, 0.0),
    )
    fig.tight_layout(rect=[0, 0.10, 1, 1])
    return fig
