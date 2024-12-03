from typing import Sequence

import seaborn as sns

from bayesflow.simulators import Simulator
from .pairs_samples import pairs_samples


def pairs_prior(
    simulator: Simulator,
    variable_names: Sequence[str] | str = None,
    num_samples: int = 2000,
    height: float = 2.5,
    color: str | tuple = "#132a70",
    **kwargs,
) -> sns.PairGrid:
    """Creates pair-plots for a given joint prior.

    Parameters
    ----------
    simulator      : bayesflow.simulations.Simulator
        The simulator object which can take an integer argument and generate random draws.
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

    return pairs_samples(
        samples, context="Prior", height=height, color=color, param_names=variable_names, render=True, **kwargs
    )
