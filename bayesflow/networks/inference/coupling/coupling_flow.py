from typing import Any

import keras

from bayesflow.masking import MaskHandler
from bayesflow.types import Tensor
from bayesflow.utils import find_permutation, weighted_mean
from bayesflow.utils.serialization import serializable, serialize

from .actnorm import ActNorm
from .layers import DualCoupling
from .permutations import FixedPermutation

from ...inference import InferenceNetwork


@serializable("bayesflow.networks")
class CouplingFlow(InferenceNetwork):
    """Coupling-based normalizing flow for simulation-based inference.

    Constructs a deep invertible architecture composed of multiple layers,
    including ActNorm, learned permutations, and dual coupling layers.
    Incorporates ideas from [1-5] and can be instantiate either an affine
    coupling flow or a spline flow.

    The specific transformation applied in the coupling layers is determined by
    *transform*, while the subnet type can be either an MLP or another callable
    architecture specified by *subnet*.  If *use_actnorm* is ``True``, an ActNorm
    layer is applied before each coupling layer.

    The model can be initialised with a base distribution, such as a standard
    normal, for density estimation.  It can also use more flexible distributions,
    e.g., GMMs for highly multimodal, low-dimensional distributions or
    Multivariate Student-*t* for heavy-tailed distributions.

    Note: Using ``transform="spline"`` produces more flexible estimators and
    generally needs ``depth < 6``. However, splines can be less stable than the
    default "affine" transform. Consider increasing the default domain of the spline
    if you see numerical instabilities (e.g., ``nan`` loss) like so:

    ``CouplingFlow(transform_kwargs={"default_domain": (-5., 5., -5., 5.)})``

    A ``fixed_target_mask`` (1 = inferred, 0 = fixed) lets the flow train on targets
    with a varying number of dimensions. Fixed dimensions are passed without being
    changed and are excluded from density computation. They still condition the
    inferred dimensions. During sampling, they are filled with ``fixed_target_value``
    (with zeros if it is not specified). Currently not supported for
    ``permutation="orthogonal"``.

    Parameters
    ----------
    subnet : str or type, optional
        Architecture for the transformation network.  Can be ``"mlp"``, a custom
        network class, or a ``Layer`` object, e.g.,
        ``bayesflow.networks.MLP(widths=[32, 32])``.  Default is ``"mlp"``.
    depth : int, optional
        The number of invertible layers in the model.  Default is 6.
    transform : str, optional
        The type of transformation used in the coupling layers.  Can be
        ``"affine"`` (:class:`~bayesflow.networks.inference.coupling.transforms.AffineTransform`)
        or ``"spline"``
        (:class:`~bayesflow.networks.inference.coupling.transforms.SplineTransform`).
        Default is ``"affine"``.
    permutation : str or None, optional
        The type of permutation applied between layers.  Can be ``"orthogonal"``
        (:class:`~bayesflow.networks.inference.coupling.permutations.OrthogonalPermutation`),
        ``"random"``
        (:class:`~bayesflow.networks.inference.coupling.permutations.RandomPermutation`),
        ``"swap"`` (:class:`~bayesflow.networks.inference.coupling.permutations.Swap`),
        or ``None`` (no permutation).  Default is ``"random"``.
    use_actnorm : bool, optional
        Whether to apply :class:`~bayesflow.networks.inference.coupling.ActNorm`
        before each coupling layer.  Default is ``True``.
    base_distribution : str or ~bayesflow.distributions.Distribution, optional
        The base probability distribution from which samples are drawn.
        Accepts the string shortcuts ``"normal"``
        (:class:`~bayesflow.distributions.DiagonalNormal`) and ``"student_t"``
        (:class:`~bayesflow.distributions.DiagonalStudentT`), or an explicit
        :class:`~bayesflow.distributions.Distribution` instance for full
        control, e.g.::
            base_distribution=bf.distributions.DiagonalStudentT(df=10.0)
        Default is ``"normal"``.
    subnet_kwargs : dict[str, Any], optional
        Keyword arguments forwarded to the subnet (e.g., MLP) constructor within
        each coupling layer, such as hidden sizes or activation choices.
    transform_kwargs : dict[str, Any], optional
        Keyword arguments forwarded to the constructor of the transform selected
        by *transform* (e.g., ``bins`` for splines).
    mask_handler : ~bayesflow.masking.MaskHandler, optional
        Handler used to generate and route masks. If ``None``, a default mask handler is created.
        Note: the coupling flow currently only supports ``fixed_target_mask``.
        Only ``fixed_target_prob`` can be specified for the mask handler.
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
        base_distribution: str | keras.Layer = "normal",
        subnet_kwargs: dict[str, Any] = None,
        transform_kwargs: dict[str, Any] = None,
        mask_handler: MaskHandler | None = None,
        **kwargs,
    ):
        super().__init__(base_distribution=base_distribution, mask_handler=mask_handler, **kwargs)

        self.subnet = subnet
        self.depth = depth
        self.transform = transform
        self.permutation = permutation
        self.use_actnorm = use_actnorm

        self.invertible_layers = []
        for _ in range(depth):
            if use_actnorm:
                self.invertible_layers.append(ActNorm())

            if (p := find_permutation(permutation)) is not None:
                self.invertible_layers.append(p)

            self.invertible_layers.append(
                DualCoupling(subnet, transform, subnet_kwargs=subnet_kwargs, transform_kwargs=transform_kwargs)
            )

        # We only need to do this for coupling flows, since we do not serialize invertible layers
        self.subnet_kwargs = subnet_kwargs
        self.transform_kwargs = transform_kwargs

    # noinspection PyMethodOverriding
    def build(self, xz_shape, conditions_shape=None):
        if self.built:
            # building when the network is already built can cause issues with serialization
            # see https://github.com/keras-team/keras/issues/21147
            return

        for layer in self.invertible_layers:
            layer.build(xz_shape=xz_shape, conditions_shape=conditions_shape)

        self.base_distribution.build(xz_shape)

    def get_config(self):
        base_config = super().get_config()

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

    def _trace_permutations(self, mask: Tensor | None) -> list[Tensor | None]:
        """Return a target mask or value in the order each invertible layer receives it, then in latent order."""
        if mask is None:
            return [None] * (len(self.invertible_layers) + 1)
        masks = [mask]
        for layer in self.invertible_layers:
            if isinstance(layer, FixedPermutation):
                mask = keras.ops.take(mask, layer.forward_indices, axis=-1)
            masks.append(mask)
        return masks

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        self.mask_handler.reject_structural_masks(kwargs, reason="coupling layers pass no masks to their subnets")
        masks = self.mask_handler.resolve(kwargs, targets=x, stage="training" if training else "inference")
        # latent_mask is the fixed_target_mask after permutation (and lives in z-space)
        *layer_masks, latent_mask = self._trace_permutations(masks.fixed_target_mask)

        z = x
        log_det = keras.ops.zeros(keras.ops.shape(x)[:-1])
        for layer, mask in zip(self.invertible_layers, layer_masks):
            z, det = layer(z, conditions=conditions, inverse=False, training=training, target_mask=mask)
            log_det += det

        if density:
            log_density_latent = self.base_distribution.log_prob(z, mask=latent_mask)
            log_density = log_density_latent + log_det
            return z, log_density

        return z

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        self.mask_handler.reject_structural_masks(kwargs, reason="coupling layers pass no masks to their subnets")
        masks = self.mask_handler.resolve(kwargs, targets=z)
        *layer_masks, latent_mask = self._trace_permutations(masks.fixed_target_mask)

        # fixed dims take their fixed value or zero when there is none (see flow matching)
        fixed_value = self._trace_permutations(masks.fixed_target_value)[-1]
        z = self.mask_handler.maybe_apply(z, latent_mask, fixed_value)

        x = z
        log_det = keras.ops.zeros(keras.ops.shape(z)[:-1])
        for layer, mask in reversed(list(zip(self.invertible_layers, layer_masks))):
            x, det = layer(x, conditions=conditions, inverse=True, training=training, target_mask=mask)
            log_det += det

        if density:
            log_prob = self.base_distribution.log_prob(z, mask=latent_mask)
            log_density = log_prob - log_det
            return x, log_density

        return x

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, sample_weight: Tensor = None, stage: str = "training", **kwargs
    ) -> dict[str, Tensor]:
        _, log_density = self(
            x, conditions=conditions, inverse=False, density=True, training=stage == "training", **kwargs
        )
        loss = weighted_mean(-log_density, sample_weight)

        return {"loss": loss}
