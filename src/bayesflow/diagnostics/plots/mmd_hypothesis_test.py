import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from keras import ops


def mmd_hypothesis_test(
    mmd_null: np.ndarray,
    mmd_observed: float = None,
    alpha_level: float = 0.05,
    null_color: str | tuple = "#132a70",
    observed_color: str | tuple = "red",
    alpha_color: str | tuple = "orange",
    truncate_v_lines_at_kde: bool = False,
    x_min: float = None,
    x_max: float = None,
    bw_factor: float = 1.5,
):
    """

    Parameters
    ----------
    mmd_null       : np.ndarray
        The samples from the MMD sampling distribution under the null hypothesis "the model is well-specified"
    mmd_observed   : float
        The observed MMD value
    alpha_level    : float, optional, default: 0.05
        The rejection probability (type I error)
    null_color     : str or tuple, optional, default: (0.16407, 0.020171, 0.577478)
        The color of the H0 sampling distribution
    observed_color : str or tuple, optional, default: "red"
        The color of the observed MMD
    alpha_color    : str or tuple, optional, default: "orange"
        The color of the rejection area
    truncate_v_lines_at_kde: bool, optional, default: False
        true: cut off the vlines at the kde
        false: continue kde lines across the plot
    x_min           : float, optional, default: None
        The lower x-axis limit
    x_max           : float, optional, default: None
        The upper x-axis limit
    bw_factor      : float, optional, default: 1.5
        bandwidth (aka. smoothing parameter) of the kernel density estimate

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    """

    def draw_v_line_to_kde(x, kde_object, color, label=None, **kwargs):
        kde_x, kde_y = kde_object.lines[0].get_data()
        idx = ops.argmin(ops.abs(kde_x - x))
        plt.vlines(x=x, ymin=0, ymax=kde_y[idx], color=color, linewidth=3, label=label, **kwargs)

    def fill_area_under_kde(kde_object, x_start, x_end=None, **kwargs):
        kde_x, kde_y = kde_object.lines[0].get_data()
        if x_end is not None:
            plt.fill_between(kde_x, kde_y, where=(kde_x >= x_start) & (kde_x <= x_end), interpolate=True, **kwargs)
        else:
            plt.fill_between(kde_x, kde_y, where=(kde_x >= x_start), interpolate=True, **kwargs)

    f = plt.figure(figsize=(8, 4))

    kde = sns.kdeplot(mmd_null, fill=False, linewidth=0, bw_adjust=bw_factor)
    sns.kdeplot(mmd_null, fill=True, alpha=0.12, color=null_color, bw_adjust=bw_factor)

    if truncate_v_lines_at_kde:
        draw_v_line_to_kde(x=mmd_observed, kde_object=kde, color=observed_color, label=r"Observed data")
    else:
        plt.vlines(
            x=mmd_observed,
            ymin=0,
            ymax=plt.gca().get_ylim()[1],
            color=observed_color,
            linewidth=3,
            label=r"Observed data",
        )

    mmd_critical = ops.quantile(mmd_null, 1 - alpha_level)
    fill_area_under_kde(
        kde, mmd_critical, color=alpha_color, alpha=0.5, label=rf"{int(alpha_level * 100)}% rejection area"
    )

    if truncate_v_lines_at_kde:
        draw_v_line_to_kde(x=mmd_critical, kde_object=kde, color=alpha_color)
    else:
        plt.vlines(x=mmd_critical, color=alpha_color, linewidth=3, ymin=0, ymax=plt.gca().get_ylim()[1])

    sns.kdeplot(mmd_null, fill=False, linewidth=3, color=null_color, label=r"$H_0$", bw_adjust=bw_factor)

    plt.xlabel(r"MMD", fontsize=20)
    plt.ylabel("")
    plt.yticks([])
    plt.xlim(x_min, x_max)
    plt.tick_params(axis="both", which="major", labelsize=16)

    plt.legend(fontsize=20)
    sns.despine()

    return f
