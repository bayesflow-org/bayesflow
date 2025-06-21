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
    A generic helper function to plot the losses of a series of training epochs and runs.

    Parameters
    ----------

    history     : keras.src.callbacks.History
        History object as returned by `keras.Model.fit`.
    train_key   : str, optional, default: "loss"
        The training loss key to look for in the history
    val_key     : str, optional, default: "val_loss"
        The validation loss key to look for in the history
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
    AssertionError
        If the number of columns in ``train_losses`` does not match the
        number of columns in ``val_losses``.
    """

    train_losses = history.history.get(train_key)
    val_losses = history.history.get(val_key)

    train_losses = pd.DataFrame(np.array(train_losses))
    val_losses = pd.DataFrame(np.array(val_losses)) if val_losses is not None else None

    # Determine the number of rows for plot
    num_row = len(train_losses.columns)

    # Initialize figure
    fig, axes = make_figure(num_row=num_row, num_col=1, figsize=(16, int(4 * num_row)) if figsize is None else figsize)

    # Get the number of steps as an array
    train_step_index = np.arange(1, len(train_losses) + 1)
    if val_losses is not None:
        val_step = int(np.floor(len(train_losses) / len(val_losses)))
        val_step_index = train_step_index[(val_step - 1) :: val_step]

        # If unequal length due to some reason, attempt a fix
        if val_step_index.shape[0] > val_losses.shape[0]:
            val_step_index = val_step_index[: val_losses.shape[0]]

    # Loop through loss entries and populate plot
    for i, ax in enumerate(axes.flat):
        if smoothing_factor > 0:
            # plot unsmoothed train loss
            ax.plot(
                train_step_index, train_losses.iloc[:, 0], color=train_color, lw=lw_train, alpha=0.3, label="Training"
            )

            # plot smoothed train loss
            smoothed_train_loss = train_losses.iloc[:, 0].ewm(alpha=1.0 - smoothing_factor, adjust=True).mean()
            ax.plot(
                train_step_index,
                smoothed_train_loss,
                color=train_color,
                lw=lw_train,
                alpha=0.8,
                label="Training (Moving Average)",
            )
        else:
            # Plot unsmoothed train loss
            ax.plot(
                train_step_index, train_losses.iloc[:, 0], color=train_color, lw=lw_train, alpha=0.8, label="Training"
            )

        # Only plot if we actually have validation losses and a color assigned
        if val_losses is not None and val_color is not None:
            alpha_unsmoothed = 0.3 if smoothing_factor > 0 else 0.8

            # Plot unsmoothed val loss
            ax.plot(
                val_step_index,
                val_losses.iloc[:, 0],
                color=val_color,
                lw=lw_val,
                alpha=alpha_unsmoothed,
                linestyle="--",
                marker=val_marker,
                markersize=val_marker_size,
                label="Validation",
            )

            # if requested, plot a second, smoothed curve
            if smoothing_factor > 0:
                smoothed_val_loss = val_losses.iloc[:, 0].ewm(alpha=1.0 - smoothing_factor, adjust=True).mean()
                ax.plot(
                    val_step_index,
                    smoothed_val_loss,
                    color=val_color,
                    linestyle="--",
                    lw=lw_val,
                    alpha=0.8,
                    label="Validation (Moving Average)",
                )

        # rest of the styling
        sns.despine(ax=ax)
        ax.grid(alpha=grid_alpha)
        ax.set_xlim(train_step_index[0], train_step_index[-1])

        # legend only if there's at least one validation curve or smoothing was on
        if val_losses is not None or smoothing_factor > 0:
            ax.legend(fontsize=legend_fontsize)

    # Add labels, titles, and set font sizes
    add_titles_and_labels(
        axes=axes,
        num_row=num_row,
        num_col=1,
        title=["Loss Trajectory"],
        xlabel="Training epoch #",
        ylabel="Value",
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )

    fig.tight_layout()
    return fig
