from typing import Literal
from collections.abc import Sequence

import keras

from bayesflow.types import Shape, Tensor
from bayesflow.utils import layer_kwargs, find_distribution
from bayesflow.utils.decorators import allow_batch_size
from bayesflow.utils.serialization import serializable


@serializable("bayesflow.networks")
class InferenceNetwork(keras.Layer):
    def __init__(
        self,
        base_distribution: Literal["normal", "student", "mixture"] | keras.Layer = "normal",
        *,
        metrics: Sequence[keras.Metric] | None = None,
        **kwargs,
    ):
        """
        Constructs an inference network using a specified base distribution and optional custom metrics.
        Use this interface for custom inference networks.

        Parameters
        ----------
        base_distribution : Literal["normal", "student", "mixture"] or keras.Layer
            Name or the actual base distribution to use. Passed to `find_distribution` to
            obtain the corresponding distribution object.
        metrics : Sequence[keras.Metric] or None, optional
            Sequence of custom Keras Metric instances to compute during training
            and evaluation. If `None`, no custom metrics are used.
        **kwargs
            Additional keyword arguments forwarded to the `keras.Layer` constructor.
        """
        self.custom_metrics = metrics
        super().__init__(**layer_kwargs(kwargs))
        self.base_distribution = find_distribution(base_distribution)

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
    def sample(self, batch_shape: Shape, conditions: Tensor = None, **kwargs) -> Tensor:
        samples = self.base_distribution.sample(batch_shape)
        samples = self(samples, conditions=conditions, inverse=True, density=False, **kwargs)
        return samples

    def log_prob(self, samples: Tensor, conditions: Tensor = None, **kwargs) -> Tensor:
        _, log_density = self(samples, conditions=conditions, inverse=False, density=True, **kwargs)
        return log_density

    def compute_metrics(
        self, x: Tensor, conditions: Tensor = None, sample_weight: Tensor = None, stage: str = "training"
    ) -> dict[str, Tensor]:
        if not self.built:
            xz_shape = keras.ops.shape(x)
            conditions_shape = None if conditions is None else keras.ops.shape(conditions)
            self.build(xz_shape, conditions_shape=conditions_shape)

        metrics = {}

        if stage != "training" and any(self.metrics):
            # compute sample-based metrics
            samples = self.sample(batch_shape=(keras.ops.shape(x)[0],), conditions=conditions)

            for metric in self.metrics:
                metrics[metric.name] = metric(samples, x)

        return metrics
