import logging
import seaborn as sns
import pandas as pd


def plot_distribution_2d(
    samples,
    context: str = None,
    height: float = 2.5,
    color: str | tuple = "#8f2727",
    alpha: float = 0.9,
    n_params: int = None,
    param_names: list = None,
    render: bool = True,
    **kwargs
):
    """
    A more flexible pairplot function for multiple distributions based upon collected samples.

    Parameters
    ----------
    samples     : np.ndarray or tf.Tensor of shape (n_sim, n_params)
        Sample draws from any dataset
    context     : str
        The context that the sample represents
    height      : float, optional, default: 2.5
        The height of the pair plot
    color       : str, optional, default : '#8f2727'
        The color of the plot
    alpha       : float in [0, 1], optonal, default: 0.9
        The opacity of the plot
    n_params     : int, optional, default: None
        The number of parameters in the collection of distributions
    param_names : list or None, optional, default: None
        The parameter names for nice plot titles. Inferred if None
    render      : bool, optional, default: True
        The boolean that determines whether to render the plot visually. If true, then the plot will render; otherwise, the plot will go through further steps for postprocessing
    **kwargs    : dict, optional
        Additional keyword arguments passed to the sns.PairGrid constructor
    """
    # Get latent dimensions
    dim = samples.shape[-1]

    # Get number of params
    if n_params is None:
        n_params = dim

    # Generate context if there is none
    if context is None:
        context = "Generic"

    # Generate titles
    if param_names is None:
        titles = [f"{context} Param. {i}" for i in range(1, dim + 1)]
    else:
        titles = [f"{context} {p}" for p in param_names]
    
    # Convert samples to pd.DataFrame
    data_to_plot = pd.DataFrame(samples, columns=titles)

    # Generate plots
    g = sns.PairGrid(data_to_plot, height=height, **kwargs)

    g.map_diag(sns.histplot, fill=True, color=color, alpha=alpha, kde=True)

    # Incorporate exceptions for generating KDE plots
    try: 
        g.map_lower(sns.kdeplot, fill=True, color=color, alpha=alpha)
    except Exception as e:
        logging.warning("KDE failed due to the following exception:\n" + repr(e) + "\nSubstituting scatter plot.")
        g.map_lower(sns.scatterplot, alpha=0.6, s=40, edgecolor="k", color=color)
    
    g.map_upper(sns.scatterplot, alpha=0.6, s=40, edgecolor="k", color=color)

    if render:
        # Generate grids
        for i in range(dim):
            for j in range(dim):
                g.axes[i, j].grid(alpha=0.5)
        
        # Return figure
        g.tight_layout()
        return g
    else:
        return g
