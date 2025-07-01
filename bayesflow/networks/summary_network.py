import keras
from keras.src.utils import python_utils
from collections.abc import Sequence

from bayesflow.metrics.functional import maximum_mean_discrepancy
from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs, find_distribution
from bayesflow.utils.decorators import sanitize_input_shape
from bayesflow.utils.serialization import serializable, serialize


@serializable("bayesflow.networks")
class SummaryNetwork(keras.Layer):
    """
    Builds a summary network with an optional base distribution and custom metrics. Use this class
    as an interface for custom summary networks.

    Important
    ---------
    If a base distribution is passed, the summary outputs will be optimized to follow
    that distribution, as described in [1].

    References
    ----------
    [1] Schmitt, M., Bürkner, P. C., Köthe, U., & Radev, S. T. (2023).
        Detecting model misspecification in amortized Bayesian inference with neural networks.
        In DAGM German Conference on Pattern Recognition (pp. 541-557).
        Cham: Springer Nature Switzerland.
    """

    def __init__(self, base_distribution: str = None, *, metrics: Sequence[keras.Metric] | None = None, **kwargs):
        """
        Creates the network with provided arguments. Optional user-supplied metrics will be stored
        in a `custom_metrics` attribute. A special `metrics` attribute will be created internally by `keras.Layer`.

        Parameters
        ----------
        base_distribution : str or None, default None
            Name of the base distribution to use. If `None`, a default distribution
            is chosen. Passed to `find_distribution` to obtain the corresponding
            distribution object.
        metrics : Sequence[keras.Metric] or None, optional
            Sequence of custom Keras Metric instances to compute during training
            and evaluation. If `None`, no custom metrics are used.
        **kwargs
            Additional keyword arguments forwarded to the `keras.Layer` constructor.
        """
        super().__init__(**layer_kwargs(kwargs))
        self.custom_metrics = metrics
        self.base_distribution = find_distribution(base_distribution)

    @sanitize_input_shape
    def build(self, input_shape):
        x = keras.ops.zeros(input_shape)
        z = self.call(x)

        if self.base_distribution is not None and not self.base_distribution.built:
            self.base_distribution.build(keras.ops.shape(z))

    @sanitize_input_shape
    def compute_output_shape(self, input_shape):
        return keras.ops.shape(self.call(keras.ops.zeros(input_shape)))

    def call(self, x: Tensor, **kwargs) -> Tensor:
        """
        :param x: Tensor of shape (batch_size, set_size, input_dim)

        :param kwargs: Additional keyword arguments.

        :return: Tensor of shape (batch_size, output_dim)
        """
        raise NotImplementedError

    def compute_metrics(self, x: Tensor, stage: str = "training", **kwargs) -> dict[str, Tensor]:
        outputs = self(x, training=stage == "training")

        metrics = {"outputs": outputs}

        if self.base_distribution is not None:
            samples = self.base_distribution.sample((keras.ops.shape(x)[0],))
            mmd = maximum_mean_discrepancy(outputs, samples)
            metrics["loss"] = keras.ops.mean(mmd)

            if stage != "training":
                # compute sample-based validation metrics
                for metric in self.metrics:
                    metrics[metric.name] = metric(outputs, samples)

        return metrics

    @python_utils.default
    def get_config(self):
        base_config = super().get_config()

        config = {"metrics": self.custom_metrics, "base_distribution": self.base_distribution}
        return base_config | serialize(config)
