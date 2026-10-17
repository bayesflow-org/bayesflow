import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import layer_kwargs
from bayesflow.utils.serialization import serializable, deserialize


@serializable("bayesflow.distributions")
class Distribution(keras.Layer):
    def __init__(self, **kwargs):
        super().__init__(**layer_kwargs(kwargs))

    def call(self, samples: Tensor) -> Tensor:
        return keras.ops.exp(self.log_prob(samples))

    def log_prob(
        self,
        samples: Tensor,
        *,
        normalize: bool = True,
        mask: Tensor | None = None,
    ) -> Tensor:
        """Evaluate a (possibly marginal) log-density.

        Parameters
        ----------
        samples : Tensor
            Samples with shape ``batch_shape + (event_dim,)``.
        normalize : bool, optional
            Whether to include normalization constants.
        mask : Tensor, optional
            Binary event mask broadcastable to ``samples``. A value of one
            keeps a coordinate in the density and zero marginalizes it out.
            Distributions whose marginals cannot be evaluated independently
            must reject non-``None`` masks explicitly.
        """
        raise NotImplementedError

    def sample(self, batch_shape: Shape, seed: int | keras.random.SeedGenerator | None = None) -> Tensor:
        """Draw samples from the distribution.

        Parameters
        ----------
        batch_shape : Shape
            The desired sample batch shape (tuple of ints), not including the
            event dimension.
        seed : int, keras.random.SeedGenerator, or None, optional
            Seed for reproducible sampling. An integer is converted to a
            ``keras.random.SeedGenerator`` and shared across all random draws in
            the call. A ``SeedGenerator`` is passed through as-is, advancing its
            state with each use. If ``None`` (default), the instance seed
            generator is used.

        Returns
        -------
        Tensor
            Samples with shape ``batch_shape + (event_dim,)``.
        """
        raise NotImplementedError

    def compute_output_shape(self, input_shape: Shape) -> Shape:
        return keras.ops.shape(self.sample(input_shape[0:1]))

    @classmethod
    def from_config(cls, config, custom_objects=None):
        # remove the legacy seed_generator arg
        config = {key: value for key, value in config.items() if key != "seed_generator"}

        return cls(**deserialize(config, custom_objects=custom_objects))
