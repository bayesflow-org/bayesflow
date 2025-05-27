import jax
import keras

from bayesflow.utils import filter_kwargs


class JAXApproximator(keras.Model):
    # noinspection PyMethodOverriding
    def compute_metrics(self, *args, **kwargs) -> dict[str, jax.Array]:
        # implemented by each respective architecture
        raise NotImplementedError

    def stateless_compute_metrics(
        self,
        trainable_variables: any,
        non_trainable_variables: any,
        metrics_variables: any,
        data: dict[str, any],
        stage: str = "training",
    ) -> (jax.Array, tuple):
        """
        Things we do for jax:
        1. Accept trainable variables as the first argument
            (can be at any position as indicated by the argnum parameter
             in autograd, but needs to be an explicit arg)
        2. Accept, potentially modify, and return other state variables
        3. Return just the loss tensor as the first value
        4. Return all other values in a tuple as the second value

        This ensures:
        1. The function is stateless
        2. The function can be differentiated with jax autograd
        """
        state_mapping = []
        state_mapping.extend(zip(self.trainable_variables, trainable_variables))
        state_mapping.extend(zip(self.non_trainable_variables, non_trainable_variables))
        state_mapping.extend(zip(self.metrics_variables, metrics_variables))

        # perform a stateless call to compute_metrics
        with keras.StatelessScope(state_mapping) as scope:
            kwargs = filter_kwargs(data | {"stage": stage}, self.compute_metrics)
            metrics = self.compute_metrics(**kwargs)

        # update variables
        non_trainable_variables = [scope.get_current_value(v) for v in self.non_trainable_variables]
        metrics_variables = [scope.get_current_value(v) for v in self.metrics_variables]

        return metrics["loss"], (metrics, non_trainable_variables, metrics_variables)

    def stateless_test_step(self, state: tuple, data: dict[str, any]) -> (dict[str, jax.Array], tuple):
        trainable_variables, non_trainable_variables, metrics_variables = state

        loss, aux = self.stateless_compute_metrics(
            trainable_variables, non_trainable_variables, metrics_variables, data=data, stage="validation"
        )
        metrics, non_trainable_variables, metrics_variables = aux

        metrics_variables = self._update_metrics(loss, metrics_variables)

        state = trainable_variables, non_trainable_variables, metrics_variables
        return metrics, state

    def stateless_train_step(self, state: tuple, data: dict[str, any]) -> (dict[str, jax.Array], tuple):
        trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables = state

        grad_fn = jax.value_and_grad(self.stateless_compute_metrics, has_aux=True)

        (loss, aux), grads = grad_fn(
            trainable_variables, non_trainable_variables, metrics_variables, data=data, stage="training"
        )
        metrics, non_trainable_variables, metrics_variables = aux

        trainable_variables, optimizer_variables = self.optimizer.stateless_apply(
            optimizer_variables, grads, trainable_variables
        )

        metrics_variables = self._update_metrics(loss, metrics_variables)

        state = trainable_variables, non_trainable_variables, optimizer_variables, metrics_variables
        return metrics, state

    def test_step(self, *args, **kwargs):
        return self.stateless_test_step(*args, **kwargs)

    def train_step(self, *args, **kwargs):
        return self.stateless_train_step(*args, **kwargs)

    def _update_metrics(self, loss: jax.Array, metrics_variables: any) -> any:
        # update the loss progress bar, and possibly metrics variables along with it
        state_mapping = list(zip(self.metrics_variables, metrics_variables))
        with keras.StatelessScope(state_mapping) as scope:
            self._loss_tracker.update_state(loss)

        metrics_variables = [scope.get_current_value(v) for v in self.metrics_variables]

        return metrics_variables
