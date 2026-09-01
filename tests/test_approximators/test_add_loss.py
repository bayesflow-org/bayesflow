import io
from contextlib import redirect_stdout

import keras
import pytest


@pytest.fixture()
def approximator_with_layer_loss(adapter):
    from bayesflow import ContinuousApproximator
    from bayesflow.networks import CouplingFlow, MLP

    class MLPWithAddedLoss(MLP):
        """MLP penalizing a weight that receives no gradient other than through ``add_loss``."""

        def build(self, input_shape):
            super().build(input_shape)
            self.probe = self.add_weight(shape=(), initializer="ones", name="probe")

        def call(self, x, training=False, **kwargs):
            x = super().call(x, training=training, **kwargs)
            self.add_loss(keras.ops.square(self.probe))
            return x

    return ContinuousApproximator(
        adapter=adapter,
        inference_network=CouplingFlow(subnet=MLPWithAddedLoss),
        summary_network=None,
    )


def test_layer_loss_reported(approximator_with_layer_loss, train_dataset, validation_dataset):
    approximator = approximator_with_layer_loss
    approximator.compile(optimizer="SGD")

    with io.StringIO() as stream:
        with redirect_stdout(stream):
            history = approximator.fit(dataset=train_dataset, validation_data=validation_dataset, epochs=2)

        output = stream.getvalue()

    assert "layer_loss" in output, "layer loss not shown in the progress bar"
    assert "layer_loss" in history.history, "layer loss not tracked in the history"
    assert "val_layer_loss" in history.history, "layer loss not tracked during validation"


def test_layer_loss_is_optimized(approximator_with_layer_loss, train_dataset):
    approximator = approximator_with_layer_loss
    approximator.compile(optimizer="SGD")

    history = approximator.fit(dataset=train_dataset, epochs=3, verbose=0)

    layer_loss = history.history["layer_loss"]
    assert layer_loss[-1] < layer_loss[0], "layer loss is not minimized, gradients do not reach it"
