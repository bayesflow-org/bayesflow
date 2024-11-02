from .plot_distribution_2d import plot_distribution_2d

from keras import backend as K


def plot_latent_space_2d(z_samples, height: float = 2.5, color="#8f2727", **kwargs):
    """Creates pair plots for the latent space learned by the inference network. Enables
    visual inspection of the latent space and whether its structure corresponds to the
    one enforced by the optimization criterion.

    Parameters
    ----------
    z_samples   : np.ndarray or tf.Tensor of shape (n_sim, n_params)
        The latent samples computed through a forward pass of the inference network.
    height      : float, optional, default: 2.5
        The height of the pair plot.
    color       : str, optional, default : '#8f2727'
        The color of the plot
    **kwargs    : dict, optional
        Additional keyword arguments passed to the sns.PairGrid constructor

    Returns
    -------
    f : plt.Figure - the figure instance for optional saving
    """

    # Try to convert z_samples, if eventually tf.Tensor is passed
    if not isinstance(z_samples, K.tf.Tensor):
        z_samples = K.constant(z_samples)

    plot_distribution_2d(z_samples, context="Latent Dim", height=height, color=color, render=True, **kwargs)
