import keras
import torch

from bayesflow.utils import filter_kwargs


class TorchApproximator(keras.Model):
    # noinspection PyMethodOverriding
    def compute_metrics(self, *args, **kwargs) -> dict[str, torch.Tensor]:
        # implemented by each respective architecture
        raise NotImplementedError

    def test_step(self, data: dict[str, any]) -> dict[str, torch.Tensor]:
        kwargs = filter_kwargs(data | {"stage": "validation"}, self.compute_metrics)
        metrics = self.compute_metrics(**kwargs)
        self._update_metrics(metrics, self._batch_size_from_data(data))
        return metrics

    def train_step(self, data: dict[str, any]) -> dict[str, torch.Tensor]:
        with torch.enable_grad():
            kwargs = filter_kwargs(data | {"stage": "training"}, self.compute_metrics)
            metrics = self.compute_metrics(**kwargs)

        loss = metrics["loss"]

        # noinspection PyUnresolvedReferences
        self.zero_grad()
        loss.backward()

        trainable_weights = self.trainable_weights[:]
        gradients = [v.value.grad for v in trainable_weights]

        # Update weights
        with torch.no_grad():
            self.optimizer.apply(gradients, trainable_weights)

        self._update_metrics(metrics, self._batch_size_from_data(data))
        return metrics

    def _update_metrics(self, metrics, sample_weight=None):
        for name, value in metrics.items():
            try:
                metric_index = self.metrics_names.index(name)
                self.metrics[metric_index].update_state(value, sample_weight=sample_weight)
            except ValueError:
                self._metrics.append(keras.metrics.Mean(name=name))
                self._metrics[-1].update_state(value, sample_weight=sample_weight)
