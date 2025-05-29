import keras
import numpy as np

from bayesflow.utils import filter_kwargs


class NumpyApproximator(keras.Model):
    # noinspection PyMethodOverriding
    def compute_metrics(self, *args, **kwargs) -> dict[str, np.ndarray]:
        # implemented by each respective architecture
        raise NotImplementedError

    def test_step(self, data: dict[str, any]) -> dict[str, np.ndarray]:
        kwargs = filter_kwargs(data | {"stage": "validation"}, self.compute_metrics)
        metrics = self.compute_metrics(**kwargs)
        self._update_metrics(metrics, self._batch_size_from_data(data))
        return metrics

    def train_step(self, data: dict[str, any]) -> dict[str, np.ndarray]:
        raise NotImplementedError("Numpy backend does not support training.")

    def _update_metrics(self, metrics, sample_weight=None):
        for name, value in metrics.items():
            try:
                metric_index = self.metrics_names.index(name)
                self.metrics[metric_index].update_state(value, sample_weight=sample_weight)
            except ValueError:
                self._metrics.append(keras.metrics.Mean(name=name))
                self._metrics[-1].update_state(value, sample_weight=sample_weight)
