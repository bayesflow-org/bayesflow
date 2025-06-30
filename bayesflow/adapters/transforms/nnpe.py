import numpy as np

from bayesflow.utils.serialization import serializable, serialize

from .elementwise_transform import ElementwiseTransform


@serializable("bayesflow.adapters")
class NNPE(ElementwiseTransform):
    """Implements noisy neural posterior estimation (NNPE) as described in [1], which adds noise following a
    spike-and-slab distribution to the training data as a mild form of data augmentation to robustify against noisy
    real-world data (see [1, 2] for benchmarks). Adds the options of automatic noise scale determination and
    dimensionwise noise application to the original implementation in [1] to provide more flexibility in dealing with
    unstandardized and heterogeneous data.

    [1] Ward, D., Cannon, P., Beaumont, M., Fasiolo, M., & Schmon, S. (2022). Robust neural posterior estimation and
    statistical model criticism. Advances in Neural Information Processing Systems, 35, 33845-33859.
    [2] Elsemüller, L., Pratz, V., von Krause, M., Voss, A., Bürkner, P. C., & Radev, S. T. (2025). Does Unsupervised
    Domain Adaptation Improve the Robustness of Amortized Bayesian Inference? A Systematic Evaluation. arXiv preprint
    arXiv:2502.04949.

    Parameters
    ----------
    spike_scale : float or np.ndarray or None, default=None
        The scale of the spike (Normal) distribution. Automatically determined if None (see “Notes” section).
        Expects a float if `per_dimension=False` or a 1D array of length `data.shape[-1]` if `per_dimension=True`.
    slab_scale : float or np.ndarray or None, default=None
        The scale of the slab (Cauchy) distribution. Automatically determined if None (see “Notes” section).
        Expects a float if `per_dimension=False` or a 1D array of length `data.shape[-1]` if `per_dimension=True`.
    per_dimension : bool, default=True
        If true, noise is applied per dimension of the last axis of the input data. If false, noise is applied globally.
        Thus, if per_dimension=True, any provided scales must be arrays with shape (n_dimensions,) and automatic
        scale determination occurs separately per dimension. If per_dimension=False, provided scales must be floats and
        automatic scale determination occurs globally. The original implementation in [1] uses global application
        (i.e., per_dimension=False), whereas dimensionwise is recommended if the data dimensions are heterogeneous.
    seed : int or None
        The seed for the random number generator. If None, a random seed is used. Used instead of np.random.Generator
        here to enable easy serialization.

    Notes
    -----
    The spike-and-slab distribution consists of a mixture of a Normal distribution (spike) and Cauchy distribution
    (slab), which are applied based on a Bernoulli random variable with p=0.5.

    The scales of the spike and slab distributions can be set manually, or they are automatically determined by scaling
    the default scales of [1] (which expect standardized data) by the standard deviation of the input data.
    For automatic determination, the standard deviation is determined either globally (if `per_dimension=False`) or per
    dimension of the last axis of the input data (if `per_dimension=True`). Note that automatic scale determination is
    applied batch-wise in the forward method, which means that determined scales can vary between batches due to varying
    standard deviations in the batch input data.

    The original implementation in [1] can be recovered by applying the following settings on standardized data:
    - `spike_scale=0.01`
    - `slab_scale=0.25`
    - `per_dimension=False`

    Examples
    --------
    >>> adapter = bf.Adapter().nnpe(["x"])
    """

    DEFAULT_SPIKE = 0.01
    DEFAULT_SLAB = 0.25

    def __init__(
        self,
        *,
        spike_scale: np.typing.ArrayLike | None = None,
        slab_scale: np.typing.ArrayLike | None = None,
        per_dimension: bool = True,
        seed: int | None = None,
    ):
        super().__init__()
        self.spike_scale = spike_scale
        self.slab_scale = slab_scale
        self.per_dimension = per_dimension
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def _resolve_scale(
        self,
        name: str,
        passed: np.typing.ArrayLike | None,
        default: float,
        data: np.ndarray,
    ) -> np.ndarray | float:
        """
        Determine spike/slab scale:
         - If `passed` is None: Automatic determination via default * std(data) (per‐dimension or global).
         - Else: Validate & cast `passed` to the correct shape/type.

        Parameters
        ----------
        name : str
            Identifier for error messages (e.g., 'spike_scale' or 'slab_scale').
        passed : float or np.ndarray or None
            User-specified scale. If None, compute as default * std(data).
            If self.per_dimension is True, this may be a 1D array of length data.shape[-1].
        default : float
            Default multiplier from [1] to apply to the standard deviation of the data.
        data : np.ndarray
            Data array to compute standard deviation from.

        Returns
        -------
        np.ndarray
            The resolved scale, either as a 0D array (if per_dimension=False) or an 1D array of length data.shape[-1]
            (if per_dimension=True).
        """

        # Get std and (expected shape) dimensionwise or globally
        if self.per_dimension:
            axes = tuple(range(data.ndim - 1))
            std = np.std(data, axis=axes)
            expected_shape = (data.shape[-1],)
        else:
            std = np.std(data)
            expected_shape = None

        # If no scale is passed, determine scale automatically given the dimensionwise or global std
        if passed is None:
            return np.array(default * std)
        # If a scale is passed, check if the passed shape matches the expected shape
        else:
            try:
                arr = np.asarray(passed, dtype=float)
            except Exception as e:
                raise TypeError(f"{name}: expected values convertible to float, got {type(passed).__name__}") from e

            if self.per_dimension:
                if arr.ndim != 1 or arr.shape != expected_shape:
                    raise ValueError(f"{name}: expected array of shape {expected_shape}, got {arr.shape}")
                return arr
            else:
                if arr.ndim != 0:
                    raise ValueError(f"{name}: expected scalar, got array of shape {arr.shape}")
                return arr

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

        spike_scale = self._resolve_scale("spike_scale", self.spike_scale, self.DEFAULT_SPIKE, data)
        slab_scale = self._resolve_scale("slab_scale", self.slab_scale, self.DEFAULT_SLAB, data)

        # Apply spike-and-slab noise
        mixture_mask = self.rng.binomial(n=1, p=0.5, size=data.shape).astype(bool)
        noise_spike = self.rng.standard_normal(size=data.shape) * spike_scale
        noise_slab = self.rng.standard_cauchy(size=data.shape) * slab_scale
        noise = np.where(mixture_mask, noise_slab, noise_spike)
        return data + noise

    def inverse(self, data: np.ndarray, **kwargs) -> np.ndarray:
        # Non-invertible transform
        return data

    def get_config(self) -> dict:
        return serialize(
            {
                "spike_scale": self.spike_scale,
                "slab_scale": self.slab_scale,
                "per_dimension": self.per_dimension,
                "seed": self.seed,
            }
        )
