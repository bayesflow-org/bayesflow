from collections.abc import Sequence

import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import layer_kwargs, find_distribution
from bayesflow.utils.decorators import allow_batch_size
from bayesflow.utils.keras_utils import resolve_seed
from bayesflow.utils.serialization import deserialize


class InferenceNetwork(keras.Layer):
    """Abstract base class for all inference networks in BayesFlow.

    An inference network learns a mapping between a data space and a latent space,
    optionally conditioned on external variables.  Concrete subclasses power the
    different approximation strategies (normalizing flows, diffusion models, flow
    matching, consistency models, …).

    Subclassing guide
    -----------------
    To implement a custom inference network, inherit from this class and override
    **at minimum** the following methods:

    ``_forward(x, conditions, density, training, **kwargs)``
        Map data *x* -> latent *z*.  When *density* is ``True`` the method must
        return a tuple ``(z, log_prob)``; otherwise just *z*.

    ``_inverse(z, conditions, density, training, **kwargs)``
        Map latent *z* -> data *x*.  Same density convention as ``_forward``.

    ``compute_metrics(x, conditions, sample_weight, stage)``
        Compute and return a ``dict[str, Tensor]`` of training metrics.  The dict
        **must** contain at least a ``"loss"`` key. This is where you implement
        the training objective for your custom inference network.

    Optionally override:

    ``build(xz_shape, conditions_shape)``
        Allocate weights that depend on the concrete tensor shapes.  Call
        ``super().build(...)`` to build the ``base_distribution`` and trigger a
        forward pass for shape inference.

    ``sample(batch_shape, conditions, **kwargs)``
        Draw samples from the learned distribution.  The default implementation
        samples from ``base_distribution`` and passes the result through
        ``_inverse``.

    ``log_prob(samples, conditions, **kwargs)``
        Evaluate the log-density of *samples* under the learned distribution.
        The default implementation calls ``_forward`` with ``density=True``.

    Parameters
    ----------
    base_distribution : str, optional
        Identifier for the base (latent) distribution, resolved via
        :func:`~bayesflow.utils.find_distribution`.  Default is ``"normal"``.
    **kwargs
        Forwarded to ``keras.Layer`` after filtering with
        :func:`~bayesflow.utils.layer_kwargs`.
    """

    # Valid mask keys to pass to subnet
    _SUBNET_MASK_KEYS = {"attention_mask", "mask"}

    def __init__(self, base_distribution: str = "normal", **kwargs):
        super().__init__(**layer_kwargs(kwargs))
        self.base_distribution = find_distribution(base_distribution)

    @staticmethod
    def _collect_mask_kwargs(keys: Sequence[str], source: dict) -> dict:
        """Extract mask kwargs from source dict.

        Looks up each key in *keys* and includes it in the result if its value
        is not ``None``.
        """
        return {key: source[key] for key in keys if source.get(key) is not None}

    def build(self, xz_shape: Shape, conditions_shape: Shape = None) -> None:
        if self.built:
            # building when the network is already built can cause issues with serialization
            # see https://github.com/keras-team/keras/issues/21147
            return

        self.base_distribution.build(xz_shape)
        x = keras.ops.zeros(xz_shape)
        conditions = keras.ops.zeros(conditions_shape) if conditions_shape is not None else None
        self.call(x, conditions, training=True)

    def call(
        self,
        xz: Tensor,
        conditions: Tensor = None,
        inverse: bool = False,
        density: bool = False,
        training: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if inverse:
            return self._inverse(xz, conditions=conditions, density=density, training=training, **kwargs)
        return self._forward(xz, conditions=conditions, density=density, training=training, **kwargs)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError

    @allow_batch_size
    def sample(
        self,
        batch_shape: Shape,
        conditions: Tensor = None,
        seed: int | keras.random.SeedGenerator | None = None,
        **kwargs,
    ) -> Tensor:
        seed = resolve_seed(seed)
        samples = self.base_distribution.sample(batch_shape, seed=seed)
        samples = self(samples, conditions=conditions, inverse=True, density=False, seed=seed, **kwargs)
        return samples

    def log_prob(self, samples: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        _, log_density = self(samples, conditions=conditions, inverse=False, density=True, **kwargs)
        return log_density

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, sample_weight: Tensor = None, stage: str = "training", **kwargs
    ) -> dict[str, Tensor]:
        raise NotImplementedError

    @classmethod
    def from_config(cls, config, custom_objects=None):
        return cls(**deserialize(config, custom_objects=custom_objects))
