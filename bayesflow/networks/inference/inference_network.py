from collections.abc import Callable, Mapping
from typing import Any

import keras

from bayesflow.masking import MaskHandler
from bayesflow.types import Shape, Tensor
from bayesflow.utils import layer_kwargs, find_distribution
from bayesflow.utils.decorators import allow_batch_size
from bayesflow.utils.keras_utils import resolve_seed
from bayesflow.utils.serialization import deserialize, serialize


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

    Every inference network owns a ``self.seed_generator``, and :meth:`sample` resolves the
    user-provided ``seed`` against it via :func:`~bayesflow.utils.keras_utils.resolve_seed` before
    passing the result on to ``base_distribution`` and ``_inverse``.  Subclasses that draw
    randomness of their own (e.g. stochastic solvers or training-time noise) must likewise resolve
    an omitted ``seed`` against ``self.seed_generator``, so that they never fall back to Keras'
    global generator (which cannot be traced under ``jax.jit``).

    Every inference network also owns a :class:`~bayesflow.masking.MaskHandler`. Implementations resolve masks once
    at the boundary of each top-level operation and reuse the returned immutable
    :class:`~bayesflow.masking.InferenceMasks` throughout solver steps or derivative
    closures. Ordinary padding and attention masks are routed separately to subnets which declare
    them explicitly. Custom implementations should follow the same boundary so mask sampling is
    neither repeated nor inserted into the backpropagation loop.

    Optionally override:

    ``sample(batch_shape, conditions, **kwargs)``
        Draw samples from the learned distribution.  The default implementation
        samples from ``base_distribution`` and passes the result through
        ``_inverse``.

    ``log_prob(samples, conditions, **kwargs)``
        Evaluate the log-density of *samples* under the learned distribution.
        The default implementation calls ``_forward`` with ``density=True``.

    Parameters
    ----------
    base_distribution : str or ~bayesflow.distributions.Distribution, optional
        The base (latent) distribution.  Accepts the string shortcuts
        ``"normal"`` (:class:`~bayesflow.distributions.DiagonalNormal`) and
        ``"student_t"`` (:class:`~bayesflow.distributions.DiagonalStudentT`),
        or an explicit :class:`~bayesflow.distributions.Distribution` instance
        for full control.  Default is ``"normal"``.
    mask_handler : ~bayesflow.masking.MaskHandler, optional
        Handler used to generate and route masks. If ``None``, a default
        :class:`~bayesflow.masking.MaskHandler` is created with the network's
        seed. A supplied handler retains its own random generator.
    seed : int, optional
        Initial seed for the network's local random generator.
    **kwargs
        Forwarded to ``keras.Layer`` after filtering with
        :func:`~bayesflow.utils.layer_kwargs`.
    """

    def __init__(
        self,
        base_distribution: str | keras.Layer = "normal",
        *,
        mask_handler: MaskHandler | None = None,
        seed: int | None = None,
        **kwargs,
    ):
        super().__init__(**layer_kwargs(kwargs))
        self.base_distribution = find_distribution(base_distribution)
        self.seed = seed
        self.seed_generator = keras.random.SeedGenerator(seed)

        if mask_handler is None:
            mask_handler = MaskHandler(seed=self.seed)
        self.mask_handler = mask_handler
        self._mask_seed_generator = mask_handler.seed_generator

    def get_config(self):
        return super().get_config() | {
            "mask_handler": serialize(self.mask_handler),
            "seed": self.seed,
        }

    def call(
        self,
        xz: Tensor,
        conditions: Tensor = None,
        inverse: bool = False,
        density: bool = False,
        training: bool = False,
        seed: int | keras.random.SeedGenerator | None = None,
        hutchinson_steps: int | None = None,
        guidance_kwargs: Mapping[str, Any] | None = None,
        **kwargs,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if hutchinson_steps is not None:
            kwargs["hutchinson_steps"] = hutchinson_steps
        if guidance_kwargs is not None:
            kwargs["guidance_kwargs"] = guidance_kwargs

        if inverse:
            return self._inverse(xz, conditions=conditions, density=density, training=training, seed=seed, **kwargs)
        return self._forward(xz, conditions=conditions, density=density, training=training, seed=seed, **kwargs)

    def _forward(
        self, x: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _inverse(
        self, z: Tensor, conditions: Tensor = None, density: bool = False, training: bool = False, **kwargs
    ) -> Tensor | tuple[Tensor, Tensor]:
        raise NotImplementedError

    def _build_subnet_and_projector(
        self,
        xz_shape: Shape,
        conditions_shape: Shape = None,
        units: int = None,
        name: str = "output_projector",
        has_time: bool = True,
    ) -> keras.Layer:
        """Build ``self.subnet`` and return a built projector to ``units`` outputs.

        The projector is the identity if the subnet already outputs ``units`` features (default: the target
        dimension), and a zero-bias dense layer named ``name`` otherwise. The subnet input is
        ``(target, time, conditions)`` when ``has_time`` is true and ``(target, conditions)`` otherwise.
        """
        units = xz_shape[-1] if units is None else units
        if has_time:
            input_shape = (xz_shape, (xz_shape[0], 1), conditions_shape)
        else:
            input_shape = (xz_shape, conditions_shape)
        self.subnet.build(input_shape)
        return self._build_projector(self.subnet.compute_output_shape(input_shape), units, name)

    @staticmethod
    def _build_projector(out_shape: Shape, units: int, name: str = "output_projector") -> keras.Layer:
        """Return a built projector to ``units`` outputs: the identity if ``out_shape`` already has ``units``
        features, and a zero-bias dense layer named ``name`` otherwise."""
        if out_shape[-1] != units:
            projector = keras.layers.Dense(units=units, bias_initializer="zeros", name=name)
        else:
            projector = keras.layers.Identity()
        projector.build(out_shape)
        return projector

    def _inverse_compositional(
        self,
        z: Tensor,
        conditions: Tensor,
        compute_prior_score: Callable = None,
        density: bool = False,
        training: bool = False,
        seed: int | keras.random.SeedGenerator | None = None,
        **kwargs,
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
        # resolved here once instead of in every subclass
        seed = resolve_seed(seed, self.seed_generator)

        samples = self.base_distribution.sample(batch_shape, seed=seed)
        if "compute_prior_score" in kwargs:
            samples = self._inverse_compositional(
                samples, conditions=conditions, inverse=True, density=False, seed=seed, **kwargs
            )
        else:
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
