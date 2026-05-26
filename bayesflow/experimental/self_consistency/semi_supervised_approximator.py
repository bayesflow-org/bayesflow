from collections.abc import Mapping, Sequence
from typing import Any

import keras

from bayesflow.adapters import Adapter
from bayesflow.types import Tensor
from bayesflow.approximators import Approximator


class SemiSupervisedApproximator(Approximator):
    def __init__(self, approximators, adapter: Adapter = None, **kwargs):
        super().__init__(**kwargs)
        self.approximators = approximators or {}
        self.adapter = adapter or Adapter()
        self._approximator_metrics = {}
        self._composite_metrics = {}

    def compile(
        self,
        *args,
        approximator_metrics: Mapping[str, Sequence[str]] | None = None,
        composite_metrics: Mapping[str, Any] | None = None,
        **kwargs,
    ):
        self._approximator_metrics = approximator_metrics or {}

        composite_metrics = composite_metrics or {}
        composite_metrics = {k: m.attach(self.approximators, self.adapter) for k, m in composite_metrics.items()}
        self._composite_metrics = composite_metrics

        return super().compile(*args, **kwargs)

    def compute_metrics(
        self, data: Mapping[str, Mapping[str, Tensor]], stage: str = "training", **kwargs
    ) -> dict[str, Tensor]:
        loss = keras.ops.zeros(())
        metrics = {}

        for key, value in data.items():
            for approx_name in self._approximator_metrics.get(key, []):
                approximator_metrics = self._compute_metrics_approximator(data=value, name=approx_name, stage=stage)
                loss += approximator_metrics["loss"]
                metrics = metrics | {f"{key}/{approx_name}/{k}": v for k, v in approximator_metrics.items()}

            if self._composite_metrics.get(key, False):
                composite_metrics = self._composite_metrics[key](value, stage=stage)
                loss += composite_metrics["loss"]
                metrics = metrics | {f"{key}/self-consistency/{k}": v for k, v in composite_metrics.items()}

        metrics["loss"] = loss

        return metrics

    def _compute_metrics_approximator(self, data: Mapping[str, Tensor], name: str, stage: str) -> Mapping[str, Tensor]:
        approximator = self.approximators.get(name)
        if not approximator or not isinstance(approximator, Approximator):
            return {}
        if approximator.adapter:
            data = approximator.adapter(data, keras=True)

        metrics = approximator.compute_metrics(**data, stage=stage)
        return metrics

    def build(self, data_shapes: Mapping[str, tuple[int] | Mapping[str, Mapping]]):
        for name, appr in self.approximators.items():
            if hasattr(appr, "built") and not appr.built:
                appr.build(data_shapes[name])
        self.built = True

    def build_from_data(self, adapted_data: Mapping[str, Mapping]):
        adapted_data = next(iter(adapted_data["data"].values()))
        # adapt data further for each approximator...
        adapted_data = self._adapt_data(adapted_data)

        self.build(keras.tree.map_structure(keras.ops.shape, adapted_data))

    def _adapt_data(self, data: Mapping[str, Any]) -> Mapping[str, Any]:
        adapted_data = {}

        for name, appr in self.approximators.items():
            if hasattr(appr, "adapter"):
                adapted_data[name] = appr.adapter(data)
            else:
                adapted_data[name] = data

        return adapted_data

    def _batch_size_from_data(self, data: Mapping[str, Any]) -> int:
        return 1
