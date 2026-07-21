from collections.abc import Mapping, Sequence
from typing import Any

import keras

from bayesflow.adapters import Adapter
from bayesflow.types import Tensor
from bayesflow.approximators import Approximator
from bayesflow.utils.serialization import serialize, serializable


@serializable("bayesflow.experimental")
class SemiSupervisedApproximator(Approximator):
    """Jointly trains different approximators on multiple datasets.

    Support semi-supervised or multi-task settings where
    different datasets (possibly labeled and unlabeled) are available and different losses
    apply to each, allowing, e.g., joint posterior and likelihood training on different datasets.
    Each training step receives a batch that is a dict-of-dicts, where
    the outer keys are dataset names and the inner dicts are the actual tensors.

    Two kinds of losses can be configured per dataset key via ``compile``:

    - **approximator_metrics**: standard supervised metrics of named sub-approximators
      (e.g. posterior trained on labeled data).
    - **composite_metrics**: higher-level losses that depend on multiple approximators,
      such as ``SelfConsistencyLoss`` (e.g. SC loss computed on unlabeled data).

    Parameters
    ----------
    approximators : dict[str, Any]
        Named sub-approximators (e.g. ``{"prior": ..., "posterior": ...,
        "likelihood": ...}``).
        Note: approximators must be of class `Approximator`, or an object with `.log_prob` method,
        or `Callable`. In the latter two cases, the approximator is only used for semi-supervised
        training.
    adapter : Adapter, optional
        Shared adapter applied to all datasets before passing to approximators and semi-supervised
        metrics. Defaults to an identity adapter.
    """

    def __init__(self, approximators, adapter: Adapter = None, **kwargs):
        super().__init__(**kwargs)
        self.approximators = approximators or {}
        self.adapter = adapter or Adapter()
        self.approximator_metrics = {}
        self.composite_metrics = {}

    def compile(
        self,
        *args,
        approximator_metrics: Mapping[str, Sequence[str]] | None = None,
        composite_metrics: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        """Configure the losses and metrics for each dataset.

        Parameters
        ----------
        approximator_metrics : Mapping[str, Sequence[str]], optional
            Maps each dataset key to a list of approximator names whose standard
            metrics could be computed on that dataset via `.compute_metrics`.
            For example, ``{"labeled": ["posterior"]}`` trains the posterior approximator
            on a dataset with key "labeled", with whatever loss is defined
            by the posterior approximator in its `.compute_metrics`.
        composite_metrics : Mapping[str, Any], optional
            Maps each dataset key to a composite metric object (e.g.
            ``SelfConsistencyLoss``). Each composite metric's ``attach`` method is
            called here so it can access the approximators and shared adapter.
            For example, ``{"unlabeled": SelfConsistencyLoss(...)}`` computes the SC
            loss on a dataset with key "labeled".
        """
        self.approximator_metrics = approximator_metrics or {}

        composite_metrics = composite_metrics or {}
        composite_metrics = {k: m.attach(self.approximators, self.adapter) for k, m in composite_metrics.items()}
        self.composite_metrics = composite_metrics

        return super().compile(*args, **kwargs)

    def compute_metrics(self, stage: str = "training", **data: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Compute the total loss and all sub-metrics for one training step.

        Parameters
        ----------
        stage : str, optional
            Current training stage (e.g., "training", "validation", "inference"). Controls
            the behavior of standardization and some metric computations (default is "training").
        **data : Mapping[str, Tensor]
            Named list of input datasets.

        Returns
        -------
        dict
            All metric values with namespaced keys like ``"labeled/posterior/loss"``
            or ``"unlabeled/self-consistency/loss"``, plus the total ``"loss"``.
        """
        loss = keras.ops.zeros(())
        metrics = {}

        for key, value in data.items():
            for approx_name in self.approximator_metrics.get(key, []):
                approximator_metrics = self._compute_metrics_approximator(data=value, name=approx_name, stage=stage)
                loss += approximator_metrics["loss"]
                metrics = metrics | {f"{key}/{approx_name}/{k}": v for k, v in approximator_metrics.items()}

            if self.composite_metrics.get(key, False):
                composite_metrics = self.composite_metrics[key](value, stage=stage)
                loss += composite_metrics["loss"]
                metrics = metrics | {f"{key}/self-consistency/{k}": v for k, v in composite_metrics.items()}

        metrics["loss"] = loss

        return metrics

    def sample(self, *, approximator: str, numpy: bool = True, **kwargs):
        """
        Generates samples from one of the approximator (specified by its name) given input conditions.
        **kwargs are passed to the approximator.sample method.
        """
        if hasattr(self.approximators[approximator], "sample"):
            samples = self.approximators[approximator].sample(**kwargs)
        else:
            raise AttributeError(f"approximator `{approximator}` does not have a `sample` method.")

        samples = self.adapter(samples, inverse=True)
        if numpy:
            samples = keras.tree.map_structure(keras.ops.convert_to_numpy, samples)

        return samples

    def _compute_metrics_approximator(self, data: Mapping[str, Tensor], name: str, stage: str) -> Mapping[str, Tensor]:
        """Run one approximator's training metrics on a single dataset.

        Applies the approximator's own adapter (if present) before calling its
        ``compute_metrics``. Returns an empty dict if the named approximator is not
        found or is not an ``Approximator`` instance.
        """
        approximator = self.approximators.get(name)
        if not approximator or not isinstance(approximator, Approximator):
            return {}
        if approximator.adapter:
            data = approximator.adapter(data, keras=True)

        metrics = approximator.compute_metrics(**data, stage=stage)
        return metrics

    def get_config(self):
        base_config = super().get_config()
        config = {
            "approximators": self.approximators,
            "adapter": self.adapter,
        }
        return base_config | serialize(config)

    def get_compile_config(self):
        base_config = super().get_compile_config() or {}
        config = {
            "approximator_metrics": self.approximator_metrics,
            "composite_metrics": self.composite_metrics,
        }
        return base_config | serialize(config)

    def build(self, data_shapes: Mapping[str, tuple[int] | Mapping[str, Mapping]]):
        for name, appr in self.approximators.items():
            if hasattr(appr, "built") and not appr.built:
                appr.build(data_shapes[name])

    def build_from_data(self, adapted_data: Mapping[str, Mapping]):
        adapted_data = next(iter(adapted_data.values()))
        adapted_data = self._adapt_data(adapted_data)

        self.build(keras.tree.map_structure(keras.ops.shape, adapted_data))

    def _adapt_data(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        adapted_data = {}

        for name, appr in self.approximators.items():
            # approximator may be a callable or an object with log_prob method, *without an adapter*
            # in that case we don't adapt the data
            if isinstance(appr, Approximator):
                adapted_data[name] = appr.adapter(data)
            else:
                adapted_data[name] = data

        return adapted_data

    def _batch_size_from_data(self, data: Mapping[str, Any]) -> int:
        # TODO: currently returns a placeholder — batch size inference is not implemented
        return 1
