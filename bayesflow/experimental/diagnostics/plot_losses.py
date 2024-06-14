import seaborn as sns

from keras import ops
from ..utils.plotutils import initialize_figure


def plot_losses(
    train_losses,
    val_losses=None,
    moving_average: bool = False,
    ma_window_fraction: float = 0.01,
    fig_size=None,
    train_color: str = "#8f2727",
    val_color: str = "black",
    lw_train: int = 2,
    lw_val: int = 3,
    grid_alpha: float = 0.5,
    legend_fontsize: int = 14,
    label_fontsize: int = 14,
    title_fontsize: int = 16,
):
    """A generic helper function to plot the losses of a series of training epochs and runs.

    Parameters
    ----------

    train_losses       : pd.DataFrame
        The (plottable) history as returned by a train_[...] method of a ``Trainer`` instance.
        Alternatively, you can just pass a data frame of validation losses instead of train losses,
        if you only want to plot the validation loss.
    val_losses         : pd.DataFrame or None, optional, default: None
        The (plottable) validation history as returned by a train_[...] method of a ``Trainer`` instance.
        If left ``None``, only train losses are plotted. Should have the same number of columns
        as ``train_losses``.
    moving_average     : bool, optional, default: False
        A flag for adding a moving average line of the train_losses.
    ma_window_fraction : int, optional, default: 0.01
        Window size for the moving average as a fraction of total training steps.
    fig_size           : tuple or None, optional, default: None
        The figure size passed to the ``matplotlib`` constructor. Inferred if ``None``
    train_color        : str, optional, default: '#8f2727'
        The color for the train loss trajectory
    val_color          : str, optional, default: black
        The color for the optional validation loss trajectory
    lw_train           : int, optional, default: 2
        The linewidth for the training loss curve
    lw_val             : int, optional, default: 3
        The linewidth for the validation loss curve
    grid_alpha         : float, optional, default 0.5
        The opacity factor for the background gridlines
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

    # Determine the number of rows for plot
    n_row = len(train_losses.columns)

    # Initialize figure
    f, axarr = initialize_figure(n_row=n_row, n_col=1, fig_size=(16, int(4 * n_row)))

    # if fig_size is None:
    #     fig_size = (16, int(4 * n_row))
    # f, axarr = plt.subplots(n_row, 1, figsize=fig_size)

    # Get the number of steps as an array
    train_step_index = ops.arange(1, len(train_losses) + 1)
    if val_losses is not None:
        val_step = int(ops.floor(len(train_losses) / len(val_losses)))
        val_step_index = train_step_index[(val_step - 1) :: val_step]

        # If unequal length due to some reason, attempt a fix
        if val_step_index.shape[0] > val_losses.shape[0]:
            val_step_index = val_step_index[: val_losses.shape[0]]

    # Loop through loss entries and populate plot
    looper = [axarr] if n_row == 1 else axarr.flat
    for i, ax in enumerate(looper):
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
        # Schmuck
        ax.set_xlabel("Training step #", fontsize=label_fontsize)
        ax.set_ylabel("Value", fontsize=label_fontsize)
        sns.despine(ax=ax)
        ax.grid(alpha=grid_alpha)
        ax.set_title(train_losses.columns[i], fontsize=title_fontsize)
        # Only add legend if there is a validation curve
        if val_losses is not None or moving_average:
            ax.legend(fontsize=legend_fontsize)
    f.tight_layout()
    return f
