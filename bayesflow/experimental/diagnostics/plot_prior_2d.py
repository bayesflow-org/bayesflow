
from .plot_distribution_2d import plot_distribution_2d


def plot_prior_2d(
    prior,
    param_names: list = None,
    n_samples: int = 2000,
    height: float = 2.5,
    color: str | tuple = "#8f2727",
    **kwargs
):
    """Creates pair-plots for a given joint prior.

    Parameters
    ----------
    prior       : callable
        The prior object which takes a single integer argument and generates random draws.
    param_names : list of str or None, optional, default None
        An optional list of strings which
    n_samples   : int, optional, default: 1000
        The number of random draws from the joint prior
    height      : float, optional, default: 2.5
        The height of the pair plot
    color       : str, optional, default : '#8f2727'
        The color of the plot
    **kwargs    : dict, optional
        Additional keyword arguments passed to the sns.PairGrid constructor

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    """

    # Generate prior draws
    prior_samples = prior(n_samples)

    # Handle dict type
    if type(prior_samples) is dict:
        prior_samples = prior_samples["prior_draws"]

    plot_distribution_2d(
        prior_samples,
        context="Prior",
        height=height,
        color=color,
        param_names=param_names,
        render=True,
        **kwargs
    )
