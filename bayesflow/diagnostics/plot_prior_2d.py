import matplotlib.pyplot as plt
from .plot_samples_2d import plot_samples_2d

from typing import Sequence


def plot_prior_2d(
    simulator,
    variable_names: Sequence[str] | str = None,
    num_samples: int = 2000,
    height: float = 2.5,
    color: str | tuple = "#132a70",
    **kwargs,
) -> plt.Figure:
    """Creates pair-plots for a given joint prior.

    Parameters
    ----------
    prior       : callable
        The prior object which takes a single integer argument and generates random draws.
    variable_names : list of str or None, optional, default None
        An optional list of strings which
    num_samples   : int, optional, default: 1000
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
    samples = simulator.sample((num_samples,))

    # Handle dict type
    if isinstance(samples, dict):
        samples = samples["theta"]

    plot_samples_2d(
        samples, context="Prior", height=height, color=color, param_names=variable_names, render=True, **kwargs
    )
