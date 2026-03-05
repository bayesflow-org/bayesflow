import keras

from bayesflow.metrics.functional import maximum_mean_discrepancy
from bayesflow.types import Tensor
from bayesflow.utils import layer_kwargs, find_distribution, filter_kwargs
from bayesflow.utils.decorators import sanitize_input_shape
from bayesflow.utils.serialization import deserialize


class SummaryNetwork(keras.Layer):
    """Abstract base class for all summary networks in BayesFlow.

    A summary network compresses variable-length (set-valued) observations into
    fixed-dimensional summary statistics that condition the inference network.

    Subclassing guide
    -----------------
    To implement a custom summary network, inherit from this class and override
    **at minimum**:

    ``call(x, **kwargs)``
        Accept a tensor of shape ``(batch_size, set_size, input_dim)`` and return
        a tensor of shape ``(batch_size, output_dim)``.

    Optionally override:

    ``compute_metrics(x, stage, **kwargs)``
        Return a ``dict[str, Tensor]`` of training metrics.  If a
        ``base_distribution`` is set the default implementation computes a
        maximum-mean-discrepancy (MMD) loss between the summary outputs and
        samples from the base distribution (useful for detecting outliers).

    ``build(input_shape)``
        Allocate weights that depend on the concrete tensor shape.  The default
        implementation runs a forward pass for shape inference and, when a
        ``base_distribution`` is configured, builds it against the output shape.

    Parameters
    ----------
    base_distribution : str or None, optional
        Identifier for an optional latent distribution used for MMD
        summary space regularization, resolved via
        :func:`~bayesflow.utils.find_distribution`.  Default is ``None``
        (no regularization).
    **kwargs
        Forwarded to ``keras.Layer`` after filtering with
        :func:`~bayesflow.utils.layer_kwargs`.
    """

    def __init__(self, base_distribution: str = None, **kwargs):
        super().__init__(**layer_kwargs(kwargs))
        self.base_distribution = find_distribution(base_distribution)

    @sanitize_input_shape
    def build(self, input_shape):
        x = keras.ops.zeros(input_shape)
        z = self.call(x)

        if self.base_distribution is not None:
            self.base_distribution.build(keras.ops.shape(z))

    @sanitize_input_shape
    def compute_output_shape(self, input_shape):
        return keras.ops.shape(self.call(keras.ops.zeros(input_shape)))

    def call(self, x: Tensor, **kwargs) -> Tensor:
        raise NotImplementedError

    def compute_metrics(self, x: Tensor, stage: str = "training", **kwargs) -> dict[str, Tensor]:
        outputs = self(x, training=stage == "training", **filter_kwargs(kwargs, self.call))

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

    @classmethod
    def from_config(cls, config, custom_objects=None):
        if hasattr(cls.get_config, "_is_default") and cls.get_config._is_default:
            return cls(**config)
        return cls(**deserialize(config, custom_objects=custom_objects))
