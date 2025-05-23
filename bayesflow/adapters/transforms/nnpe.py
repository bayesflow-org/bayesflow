import numpy as np

from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class Nnpe(ElementwiseTransform):
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
    slab_scale : float
        The scale of the slab (Cauchy) distribution.
    spike_scale : float
        The scale of the spike spike (Normal) distribution.
    seed : int or None
        The seed for the random number generator. If None, a random seed is used. Used instead of np.random.Generator
        here to enable easy serialization.

    Notes
    -----
    The spike-and-slab distribution consists of a mixture of a Cauchy (slab) and a Normal distribution (spike), which
    are applied based on a Bernoulli random variable with p=0.5.

    The default scales follow [1] and expect standardized data (e.g., via the `Standardize` adapter). It is therefore
    recommended to adapt the scales when using unstandardized training data.

    Examples
    --------
    >>> adapter = bf.Adapter().nnpe(["x"])
    """

    def __init__(self, *, slab_scale: float = 0.25, spike_scale: float = 0.01, seed: int = None):
        super().__init__()
        self.slab_scale = slab_scale
        self.spike_scale = spike_scale
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def forward(self, data: np.ndarray, stage: str = "inference", **kwargs) -> np.ndarray:
        if stage != "training":
            return data
        mixture_mask = self.rng.binomial(n=1, p=0.5, size=data.shape).astype(bool)
        noise_slab = self.rng.standard_cauchy(size=data.shape) * self.slab_scale
        noise_spike = self.rng.standard_normal(size=data.shape) * self.spike_scale
        noise = np.where(mixture_mask, noise_slab, noise_spike)
        return data + noise

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        return data

    def get_config(self) -> dict:
        return serialize({"slab_scale": self.slab_scale, "spike_scale": self.spike_scale, "seed": self.seed})
