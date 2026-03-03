import keras

from bayesflow.types import Tensor
from bayesflow.utils import (
    find_permutation,
    layer_kwargs,
    weighted_mean,
)
from bayesflow.utils.serialization import serializable, serialize

from .actnorm import ActNorm
from .couplings import DualCoupling

from ..inference_network import InferenceNetwork


@serializable("bayesflow.networks")
class CouplingFlow(InferenceNetwork):
    """Coupling-based normalizing flow for simulation-based inference.

    Constructs a deep invertible architecture composed of multiple layers,
    including ActNorm, learned permutations, and dual coupling layers.
    Incorporates ideas from [1-5].

    The specific transformation applied in the coupling layers is determined by
    *transform*, while the subnet type can be either an MLP or another callable
    architecture specified by *subnet*.  If *use_actnorm* is ``True``, an ActNorm
    layer is applied before each coupling layer.

    The model can be initialised with a base distribution, such as a standard
    normal, for density estimation.  It can also use more flexible distributions,
    e.g., GMMs for highly multimodal, low-dimensional distributions or
    Multivariate Student-*t* for heavy-tailed distributions.

    Parameters
    ----------
    subnet : str or type, optional
        Architecture for the transformation network.  Can be ``"mlp"``, a custom
        network class, or a ``Layer`` object, e.g.,
        ``bayesflow.networks.MLP(widths=[32, 32])``.  Default is ``"mlp"``.
    depth : int, optional
        The number of invertible layers in the model.  Default is 6.
    transform : str, optional
        The type of transformation used in the coupling layers, such as
        ``"affine"``.  Default is ``"affine"``.
    permutation : str or None, optional
        The type of permutation applied between layers.  Can be ``"orthogonal"``,
        ``"random"``, ``"swap"``, or ``None`` (no permutation).  Default is
        ``"random"``.
    use_actnorm : bool, optional
        Whether to apply ActNorm before each coupling layer.  Default is ``True``.
    base_distribution : str, optional
        The base probability distribution from which samples are drawn.
        Default is ``"normal"``.
    subnet_kwargs : dict[str, any], optional
        Keyword arguments forwarded to the subnet (e.g., MLP) constructor within
        each coupling layer, such as hidden sizes or activation choices.
    transform_kwargs : dict[str, any], optional
        Keyword arguments forwarded to the affine or spline transforms
        (e.g., number of bins for splines).
    **kwargs
        Additional keyword arguments passed to the base ``InferenceNetwork``.

    References
    ----------
    [1] Kingma, D. P., & Dhariwal, P. (2018). Glow: Generative flow with
        invertible 1x1 convolutions. NeurIPS, 31.
    [2] Durkan, C., et al. (2019). Neural spline flows. NeurIPS, 32.
    [3] Ardizzone, L., et al. (2020). Conditional invertible neural networks for
        diverse image-to-image translation. DAGM GCPR (pp. 373-387).
    [4] Radev, S. T., et al. (2020). BayesFlow: Learning complex stochastic
        models with invertible neural networks. IEEE TNNLS.
    [5] Alexanderson, S., & Henter, G. E. (2020). Robust model training and
        generalisation with Studentising flows. arXiv:2006.06599.
    """

    def __init__(
        self,
        subnet: str | type = "mlp",
        depth: int = 6,
        transform: str = "affine",
        permutation: str | None = "random",
        use_actnorm: bool = True,
        base_distribution: str = "normal",
        subnet_kwargs: dict[str, any] = None,
        transform_kwargs: dict[str, any] = None,
        **kwargs,
    ):
        super().__init__(base_distribution=base_distribution, **kwargs)

        self.subnet = subnet
        self.depth = depth
        self.transform = transform
        self.permutation = permutation
        self.use_actnorm = use_actnorm

        self.invertible_layers = []
        for i in range(depth):
            if use_actnorm:
                self.invertible_layers.append(ActNorm())

            if (p := find_permutation(permutation)) is not None:
                self.invertible_layers.append(p)

            self.invertible_layers.append(
                DualCoupling(subnet, transform, subnet_kwargs=subnet_kwargs, transform_kwargs=transform_kwargs)
            )

        # We only need to do this from coupling flows, since we do not serialize invertible layers
        self.subnet_kwargs = subnet_kwargs
        self.transform_kwargs = transform_kwargs

    # noinspection PyMethodOverriding
    def build(self, xz_shape, conditions_shape=None):
        for layer in self.invertible_layers:
            layer.build(xz_shape=xz_shape, conditions_shape=conditions_shape)

        self.base_distribution.build(xz_shape)

    def get_config(self):
        base_config = super().get_config()
        base_config = layer_kwargs(base_config)

        config = {
            "subnet": self.subnet,
            "depth": self.depth,
            "transform": self.transform,
            "permutation": self.permutation,
            "use_actnorm": self.use_actnorm,
            "base_distribution": self.base_distribution,
            "subnet_kwargs": self.subnet_kwargs,
            "transform_kwargs": self.transform_kwargs,
        }

        return base_config | serialize(config)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        z = x
        log_det = keras.ops.zeros(keras.ops.shape(x)[:-1])
        for layer in self.invertible_layers:
            z, det = layer(z, conditions=conditions, inverse=False, training=training)
            log_det += det

        if density:
            log_density_latent = self.base_distribution.log_prob(z)
            log_density = log_density_latent + log_det
            return z, log_density

        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        x = z
        log_det = keras.ops.zeros(keras.ops.shape(z)[:-1])
        for layer in reversed(self.invertible_layers):
            x, det = layer(x, conditions=conditions, inverse=True, training=training)
            log_det += det

        if density:
            log_prob = self.base_distribution.log_prob(z)
            log_density = log_prob - log_det
            return x, log_density

        return x

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, sample_weight: Tensor = None, stage: str = "training"
    ) -> dict[str, Tensor]:
        metrics = super().compute_metrics(x=x, conditions=conditions, sample_weight=sample_weight, stage=stage)

        z, log_density = self(x, conditions=conditions, inverse=False, density=True, training=stage == "training")
        loss = weighted_mean(-log_density, sample_weight)

        return metrics | {"loss": loss}
