import logging

import numpy as np
import seaborn as sns
import pandas as pd


def plot_samples_2d(
    samples: np.ndarray = None,
    context: str = None,
    n_params: int = None,
    param_names: list = None,
    height: float = 2.5,
    color: str | tuple = "#1a1e75",
    alpha: float = 0.9,
    render: bool = True,
    **kwargs,
):
    """
    A more flexible pair plot function for multiple distributions based upon
    collected samples.

    Parameters
    ----------
    samples     : dict[str, Tensor], default: None
        Sample draws from any dataset
    context     : str, default: None
        The context that the sample represents
    height      : float, optional, default: 2.5
        The height of the pair plot
    color       : str, optional, default : '#8f2727'
        The color of the plot
    alpha       : float in [0, 1], optional, default: 0.9
        The opacity of the plot
    n_params     : int, optional, default: None
        The number of params in the collection of distributions
    param_names : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    render      : bool, optional, default: True
        The boolean that determines whether to render the plot visually.
        If true, then the plot will render;
        otherwise, the plot will go through further steps for postprocessing.
    **kwargs    : dict, optional
        Additional keyword arguments passed to the sns.PairGrid constructor
    """
    # Get latent dimensions
    dim = samples.shape[-1]

    # Get number of params
    if n_params is None:
        n_params = dim

    # Generate parameters if there is none
    if context is None:
        context = "Default"

    # Generate titles
    if param_names is None:
        titles = [f"{context} $\\theta_{{{i}}}$" for i in range(1, dim + 1)]
    else:
        titles = [f"{context} {p}" for p in param_names]

    # Convert samples to pd.DataFrame
    data_to_plot = pd.DataFrame(samples, columns=titles)

    # Generate plots
    artist = sns.PairGrid(data_to_plot, height=height, **kwargs)

    artist.map_diag(sns.histplot, fill=True, color=color, alpha=alpha, kde=True)

    # Incorporate exceptions for generating KDE plots
    try:
        artist.map_lower(sns.kdeplot, fill=True, color=color, alpha=alpha)
    except Exception as e:
        logging.warning("KDE failed due to the following exception:\n" + repr(e) + "\nSubstituting scatter plot.")
        artist.map_lower(sns.scatterplot, alpha=0.6, s=40, edgecolor="k", color=color)

    artist.map_upper(sns.scatterplot, alpha=0.6, s=40, edgecolor="k", color=color)

    if render:
        # Generate grids
        for i in range(dim):
            for j in range(dim):
                artist.axes[i, j].grid(alpha=0.5)

        # Return figure
        artist.tight_layout()

    return artist
