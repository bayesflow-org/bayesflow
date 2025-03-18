from typing import Sequence

import keras.src.callbacks
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from ...utils.plot_utils import make_figure, add_titles_and_labels


def loss(
    history: keras.src.callbacks.History,
    train_key: str = "loss",
    val_key: str = "val_loss",
    moving_average: bool = False,
    per_training_step: bool = False,
    ma_window_fraction: float = 0.01,
    figsize: Sequence[float] = None,
    train_color: str = "#132a70",
    val_color: str = "black",
    lw_train: float = 2.0,
    lw_val: float = 3.0,
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
    moving_average     : bool, optional, default: False
        A flag for adding a moving average line of the train_losses.
    per_training_step : bool, optional, default: False
        A flag for making loss trajectory detailed (to training steps) rather than per epoch.
    ma_window_fraction : int, optional, default: 0.01
        Window size for the moving average as a fraction of total
        training steps.
    figsize            : tuple or None, optional, default: None
        The figure size passed to the ``matplotlib`` constructor.
        Inferred if ``None``
    train_color        : str, optional, default: '#8f2727'
        The color for the train loss trajectory
    val_color          : str, optional, default: black
        The color for the optional validation loss trajectory
    lw_train           : int, optional, default: 2
        The linewidth for the training loss curve
    lw_val             : int, optional, default: 3
        The linewidth for the validation loss curve
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
        # Plot train curve
        ax.plot(train_step_index, train_losses.iloc[:, i], color=train_color, lw=lw_train, alpha=0.9, label="Training")
        if moving_average and train_losses.columns[i] == "Loss":
            moving_average_window = int(train_losses.shape[0] * ma_window_fraction)
            smoothed_loss = train_losses.iloc[:, i].rolling(window=moving_average_window).mean()
            ax.plot(train_step_index, smoothed_loss, color="grey", lw=lw_train, label="Training (Moving Average)")

        # Plot optional val curve
        if val_losses is not None:
            if i < val_losses.shape[1]:
                ax.plot(
                    val_step_index,
                    val_losses.iloc[:, i],
                    linestyle="--",
                    marker="o",
                    color=val_color,
                    lw=lw_val,
                    label="Validation",
                )

        sns.despine(ax=ax)
        ax.grid(alpha=0.5)

        # Only add legend if there is a validation curve
        if val_losses is not None or moving_average:
            ax.legend(fontsize=legend_fontsize)

    # Add labels, titles, and set font sizes
    add_titles_and_labels(
        axes=axes,
        num_row=num_row,
        num_col=1,
        title=["Loss Trajectory"],
        xlabel="Training step #" if per_training_step else "Training epoch #",
        ylabel="Value",
        title_fontsize=title_fontsize,
        label_fontsize=label_fontsize,
    )

    fig.tight_layout()
    return fig
