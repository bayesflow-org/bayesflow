import numpy as np

from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class NNPE(ElementwiseTransform):
    """Implements noisy neural posterior estimation (NNPE) as described in [1], which adds noise following a
    spike-and-slab distribution to the training data as a mild form of data augmentation to robustify against noisy
    real-world data (see [1, 2] for benchmarks).

    [1] Ward, D., Cannon, P., Beaumont, M., Fasiolo, M., & Schmon, S. (2022). Robust neural posterior estimation and
    statistical model criticism. Advances in Neural Information Processing Systems, 35, 33845-33859.
    [2] Elsemüller, L., Pratz, V., von Krause, M., Voss, A., Bürkner, P. C., & Radev, S. T. (2025). Does Unsupervised
    Domain Adaptation Improve the Robustness of Amortized Bayesian Inference? A Systematic Evaluation. arXiv preprint
    arXiv:2502.04949.

    Parameters
    ----------
    spike_scale : float or None
        The scale of the spike (Normal) distribution. Automatically determined if None (see “Notes” section).
    slab_scale : float or None
        The scale of the slab (Cauchy) distribution. Automatically determined if None (see “Notes” section).
    seed : int or None
        The seed for the random number generator. If None, a random seed is used. Used instead of np.random.Generator
        here to enable easy serialization.

    Notes
    -----
    The spike-and-slab distribution consists of a mixture of a Normal distribution (spike) and Cauchy distribution
    (slab), which are applied based on a Bernoulli random variable with p=0.5.

    The scales of the spike and slab distributions can be set manually, or they are automatically determined by scaling
    the default scales of [1] (which expect standardized data) by the standard deviation of the input data.

    Examples
    --------
    >>> adapter = bf.Adapter().nnpe(["x"])
    """

    DEFAULT_SLAB = 0.25
    DEFAULT_SPIKE = 0.01

    def __init__(self, *, spike_scale: float | None = None, slab_scale: float | None = None, seed: int | None = None):
        super().__init__()
        self.spike_scale = spike_scale
        self.slab_scale = slab_scale
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def forward(self, data: np.ndarray, stage: str = "inference", **kwargs) -> np.ndarray:
        """
        Add spike‐and‐slab noise to `data` during training, using automatic scale determination if not provided (see
        “Notes” section of the class docstring for details).

        Parameters
        ----------
        data : np.ndarray
            Input array to be perturbed.
        stage : str, default='inference'
            If 'training', noise is added; else data is returned unchanged.
        **kwargs
            Unused keyword arguments.

        Returns
        -------
        np.ndarray
            Noisy data when `stage` is 'training', otherwise the original input.
        """
        if stage != "training":
            return data

        # Check data validity
        if not np.all(np.isfinite(data)):
            raise ValueError("NNPE.forward: `data` contains NaN or infinite values.")

        # Automatically determine scales if not provided
        if self.spike_scale is None or self.slab_scale is None:
            data_std = np.std(data)
        spike_scale = self.spike_scale if self.spike_scale is not None else self.DEFAULT_SPIKE * data_std
        slab_scale = self.slab_scale if self.slab_scale is not None else self.DEFAULT_SLAB * data_std

        # Apply spike-and-slab noise
        mixture_mask = self.rng.binomial(n=1, p=0.5, size=data.shape).astype(bool)
        noise_spike = self.rng.standard_normal(size=data.shape) * spike_scale
        noise_slab = self.rng.standard_cauchy(size=data.shape) * slab_scale
        noise = np.where(mixture_mask, noise_slab, noise_spike)
        return data + noise

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        """Non-invertible transform."""
        return data

    def get_config(self) -> dict:
        return serialize({"spike_scale": self.spike_scale, "slab_scale": self.slab_scale, "seed": self.seed})
