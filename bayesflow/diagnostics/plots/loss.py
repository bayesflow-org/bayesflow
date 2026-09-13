from collections.abc import Sequence

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import keras.src.callbacks

from ...utils.plot_utils import make_figure, add_titles_and_labels


def loss(
    history: keras.callbacks.History,
    train_key: str = "loss",
    val_key: str = "val_loss",
    show_components: bool = False,
    smoothing_factor: float = 0.8,
    figsize: Sequence[float] = None,
    train_color: str = "#132a70",
    val_color: str = "black",
    val_marker: str = "o",
    val_marker_size: float = 5,
    lw_train: float = 2.0,
    lw_val: float = 2.0,
    grid_alpha: float = 0.2,
    legend_fontsize: int = 14,
    label_fontsize: int = 14,
    title_fontsize: int = 16,
) -> plt.Figure:
    """
    Plot the training (and validation) loss of a series of training epochs and runs.

    Parameters
    ----------

    history     : keras.src.callbacks.History
        History object as returned by `keras.Model.fit`.
    train_key   : str, optional, default: "loss"
        The training loss key to look for in the history
    val_key     : str, optional, default: "val_loss"
        The validation loss key to look for in the history
    show_components : bool, optional, default: False
        If True, every other metric in the history (e.g., regularization losses) is
        plotted in its own panel below the total loss, with its validation counterpart
        (``"val_"`` prefix) overlaid if present.
    smoothing_factor : float, optional, default: 0.8
        If greater than zero, smooth the loss curves by applying an exponential moving average.
    figsize            : tuple or None, optional, default: None
        The figure size passed to the ``matplotlib`` constructor.
        Inferred if ``None``
    train_color        : str, optional, default: '#132a70'
        The color for the train loss trajectory
    val_color          : str, optional, default: None
        The color for the optional validation loss trajectory
    val_marker: str
        Marker style for the validation loss curve. Default is "o".
    val_marker_size: float
        Marker size for the validation loss curve. Default is 5.
    lw_train           : int, optional, default: 2
        The line width for the training loss curve
    lw_val             : int, optional, default: 2
        The line width for the validation loss curve
    grid_alpha          : float, optional, default: 0.2
        The transparency of the background grid
    legend_fontsize    : int, optional, default: 14
        The font size of the legend text
    label_fontsize     : int, optional, default: 14
        The font size of the y-label text
    title_fontsize     : int, optional, default: 16
        The font size of the title text

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ValueError
        If the loss history is not one-dimensional.
    """

    keys = [train_key]
    if show_components:
        keys += [k for k in history.history if k != train_key and not k.startswith("val_")]

    train_losses = []
    val_losses = []
    for key in keys:
        train = np.asarray(history.history[key])
        if train.ndim != 1:
            raise ValueError(f"Expected a one-dimensional history for '{key}', got shape {train.shape}.")
        train_losses.append(pd.Series(train))

        val_key_ = val_key if key == train_key else f"val_{key}"
        val = history.history.get(val_key_)
        val_losses.append(pd.Series(np.asarray(val)) if val is not None else None)

    has_val = any(v is not None for v in val_losses) and val_color is not None
    num_row = len(keys)

    fig, axes = make_figure(num_row=num_row, num_col=1, figsize=(16, int(4 * num_row)) if figsize is None else figsize)

    for ax, train, val in zip(axes.flat, train_losses, val_losses):
        train_step_index = np.arange(1, len(train) + 1)

        if smoothing_factor > 0:
            ax.plot(train_step_index, train, color=train_color, lw=lw_train, alpha=0.3, label="Training")
            smoothed_train = train.ewm(alpha=1.0 - smoothing_factor, adjust=True).mean()
            ax.plot(
                train_step_index,
                smoothed_train,
                color=train_color,
                lw=lw_train,
                alpha=0.8,
                label="Training (Moving Average)",
            )
        else:
            ax.plot(train_step_index, train, color=train_color, lw=lw_train, alpha=0.8, label="Training")

        if val is not None and val_color is not None:
            val_step = int(np.floor(len(train) / len(val)))
            val_step_index = train_step_index[(val_step - 1) :: val_step][: len(val)]

            ax.plot(
                val_step_index,
                val,
                color=val_color,
                lw=lw_val,
                alpha=0.3 if smoothing_factor > 0 else 0.8,
                linestyle="--",
                marker=val_marker,
                markersize=val_marker_size,
                label="Validation",
            )

            if smoothing_factor > 0:
                smoothed_val = val.ewm(alpha=1.0 - smoothing_factor, adjust=True).mean()
                ax.plot(
                    val_step_index,
                    smoothed_val,
                    color=val_color,
                    linestyle="--",
                    lw=lw_val,
                    alpha=0.8,
                    label="Validation (Moving Average)",
                )

        sns.despine(ax=ax)
        ax.grid(alpha=grid_alpha)
        ax.set_xlim(train_step_index[0], train_step_index[-1])

    # single legend below the figure, only if there's at least one validation curve or smoothing was on
    show_legend = has_val or smoothing_factor > 0
    if show_legend:
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(
            handles,
            labels,
            loc="lower center",
            ncol=min(len(labels), 2) if fig.get_figwidth() < 12 else len(labels),
            frameon=True,
            fontsize=legend_fontsize,
        )

    add_titles_and_labels(
        axes=axes,
        num_row=num_row,
        num_col=1,
        title=["Loss Trajectory"],
        xlabel="Training epoch #",
        ylabel=["Loss"] + keys[1:],
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )

    fig.tight_layout(rect=(0, 0.13, 1, 1) if show_legend else None)
    return fig
