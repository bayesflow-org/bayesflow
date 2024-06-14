import seaborn as sns

from keras import ops
from keras import backend as K
from ..utils.computils import simultaneous_ecdf_bands
from ..utils.plotutils import preprocess, remove_unused_axes


def plot_sbc_ecdf(
    post_samples,
    prior_samples,
    difference: bool = False,
    stacked: bool = False,
    fig_size: tuple = None,
    param_names: list = None,
    label_fontsize: int = 16,
    legend_fontsize: int = 14,
    title_fontsize: int = 18,
    tick_fontsize: int = 12,
    rank_ecdf_color: str | tuple = "#a34f4f",
    fill_color: str | tuple = "grey",
    n_row: int = None,
    n_col: int = None,
    **kwargs,
):
    """Creates the empirical CDFs for each marginal rank distribution and plots it against
    a uniform ECDF. ECDF simultaneous bands are drawn using simulations from the uniform,
    as proposed by [1].

    For models with many parameters, use `stacked=True` to obtain an idea of the overall calibration
    of a posterior approximator.

    [1] Säilynoja, T., Bürkner, P. C., & Vehtari, A. (2022). Graphical test for discrete uniformity and
    its applications in goodness-of-fit evaluation and multiple sample comparison. Statistics and Computing,
    32(2), 1-21. https://arxiv.org/abs/2103.10522

    Parameters
    ----------
    post_samples      : np.ndarray of shape (n_data_sets, n_post_draws, n_params)
        The posterior draws obtained from n_data_sets
    prior_samples     : np.ndarray of shape (n_data_sets, n_params)
        The prior draws obtained for generating n_data_sets
    difference        : bool, optional, default: False
        If `True`, plots the ECDF difference. Enables a more dynamic visualization range.
    stacked           : bool, optional, default: False
        If `True`, all ECDFs will be plotted on the same plot. If `False`, each ECDF will
        have its own subplot, similar to the behavior of `plot_sbc_histograms`.
    param_names       : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None. Only relevant if `stacked=False`.
    fig_size          : tuple or None, optional, default: None
        The figure size passed to the matplotlib constructor. Inferred if None.
    label_fontsize    : int, optional, default: 16
        The font size of the y-label and y-label texts
    legend_fontsize   : int, optional, default: 14
        The font size of the legend text
    title_fontsize    : int, optional, default: 18
        The font size of the title text. Only relevant if `stacked=False`
    tick_fontsize     : int, optional, default: 12
        The font size of the axis ticklabels
    rank_ecdf_color   : str, optional, default: '#a34f4f'
        The color to use for the rank ECDFs
    fill_color        : str, optional, default: 'grey'
        The color of the fill arguments.
    n_row             : int, optional, default: None
        The number of rows for the subplots. Dynamically determined if None.
    n_col             : int, optional, default: None
        The number of columns for the subplots. Dynamically determined if None.
    **kwargs          : dict, optional, default: {}
        Keyword arguments can be passed to control the behavior of ECDF simultaneous band computation
        through the ``ecdf_bands_kwargs`` dictionary. See `simultaneous_ecdf_bands` for keyword arguments

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving

    Raises
    ------
    ShapeError
        If there is a deviation form the expected shapes of `post_samples` and `prior_samples`.
    """

    f, ax, ax_it, n_row, n_col, n_params, param_names = preprocess(
        post_samples, prior_samples, collapse=False, fig_size=fig_size)

    # Compute fractional ranks (using broadcasting)
    post_samples = K.constant(post_samples)
    prior_samples = K.constant(prior_samples)

    # Adding an extra dimension to prior_samples using K.expand_dims
    prior_samples_expanded = K.expand_dims(prior_samples, axis=1)

    # Performing element-wise comparison
    comparison = K.less(post_samples, prior_samples_expanded)

    # Summing along the specified axis (axis=1)
    sums = K.sum(K.cast(comparison, dtype='float32'), axis=1)

    # Getting the shape of post_samples
    post_samples_shape = K.shape(post_samples)

    # Computing the ranks
    ranks = sums / K.cast(post_samples_shape[1], dtype='float32')

    # ranks = ops.sum(post_samples < prior_samples[:, np.newaxis, :], axis=1) / post_samples.shape[1]


    # Plot individual ecdf of parameters
    for j in range(ranks.shape[-1]):
        ecdf_single = ops.sort(ranks[:, j])
        xx = ecdf_single
        yy = ops.arange(1, xx.shape[-1] + 1) / float(xx.shape[-1])

        # Difference, if specified
        if difference:
            yy -= xx

        if stacked:
            if j == 0:
                ax.plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label="Rank ECDFs")
            else:
                ax.plot(xx, yy, color=rank_ecdf_color, alpha=0.95)
        else:
            ax.flat[j].plot(xx, yy, color=rank_ecdf_color, alpha=0.95, label="Rank ECDF")

    # Compute uniform ECDF and bands
    alpha, z, L, H = simultaneous_ecdf_bands(post_samples.shape[0], **kwargs.pop("ecdf_bands_kwargs", {}))

    # Difference, if specified
    if difference:
        L -= z
        H -= z
        ylab = "ECDF difference"
    else:
        ylab = "ECDF"

    # Add simultaneous bounds
    if stacked:
        titles = [None]
        axes = [ax]
    else:
        axes = ax.flat
        if param_names is None:
            titles = [f"$\\theta_{{{i}}}$" for i in range(1, n_params + 1)]
        else:
            titles = param_names

    for _ax, title in zip(axes, titles):
        _ax.fill_between(z, L, H, color=fill_color, alpha=0.2, label=rf"{int((1-alpha) * 100)}$\%$ Confidence Bands")

        # Prettify plot
        sns.despine(ax=_ax)
        _ax.grid(alpha=0.35)
        _ax.legend(fontsize=legend_fontsize)
        _ax.set_title(title, fontsize=title_fontsize)
        _ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        _ax.tick_params(axis="both", which="minor", labelsize=tick_fontsize)

    # Only add x-labels to the bottom row
    if stacked:
        bottom_row = [ax]
    else:
        bottom_row = ax if n_row == 1 else ax[-1, :]
    for _ax in bottom_row:
        _ax.set_xlabel("Fractional rank statistic", fontsize=label_fontsize)

    # Only add y-labels to right left-most row
    if n_row == 1:  # if there is only one row, the ax array is 1D
        axes[0].set_ylabel(ylab, fontsize=label_fontsize)
    else:  # if there is more than one row, the ax array is 2D
        for _ax in ax[:, 0]:
            _ax.set_ylabel(ylab, fontsize=label_fontsize)

    # Remove unused axes entirely
    remove_unused_axes(ax)

    f.tight_layout()
    return f
