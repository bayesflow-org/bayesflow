import pytest
import keras
import io
from contextlib import redirect_stdout


@pytest.fixture()
def approximator_using_add_loss(adapter):
    from bayesflow import ContinuousApproximator
    from bayesflow.networks import CouplingFlow, MLP

    class MLPAddedLoss(MLP):
        def call(self, x, training=False, **kwargs):
            x = super().call(x, training=training, **kwargs)
            self.add_loss(keras.ops.sum(x**2))
            return x

    return ContinuousApproximator(
        adapter=adapter,
        inference_network=CouplingFlow(subnet=MLPAddedLoss),
        summary_network=None,
    )


def test_layer_loss_reported(approximator_using_add_loss, train_dataset, validation_dataset):
    approximator = approximator_using_add_loss
    approximator.compile(optimizer="AdamW")
    num_epochs = 3

    # Capture ostream and train model
    with io.StringIO() as stream:
        with redirect_stdout(stream):
            approximator.fit(dataset=train_dataset, validation_data=validation_dataset, epochs=num_epochs)

        output = stream.getvalue()

    print(output)

    # check that there is a progress bar
    assert "━" in output, "no progress bar"

    # check that layer_loss is reported
    assert "layer_loss" in output, "no layer_loss"
