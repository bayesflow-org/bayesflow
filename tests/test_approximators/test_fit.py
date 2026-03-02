import re
import keras

import pytest
import io
from contextlib import redirect_stdout


from bayesflow.approximators.ensemble_approximator import EnsembleApproximator
from bayesflow.datasets.ensemble_dataset import EnsembleDataset
from bayesflow.datasets.online_dataset import OnlineDataset

from tests.utils import check_combination_simulator_adapter


@pytest.mark.skip(reason="not implemented")
def test_compile(amortizer):
    amortizer.compile(optimizer="AdamW")


@pytest.mark.skip(reason="not implemented")
def test_fit(amortizer, dataset):
    amortizer.compile(optimizer="AdamW")
    amortizer.fit(dataset)

    assert amortizer.losses is not None


def test_loss_progress_offline(approximator, train_dataset, validation_dataset):
    approximator.compile(optimizer="AdamW")
    num_epochs = 3

    if isinstance(approximator, EnsembleApproximator):
        train_dataset = EnsembleDataset(train_dataset, member_names=list(approximator.approximators.keys()))

    # Capture ostream and train model
    with io.StringIO() as stream:
        with redirect_stdout(stream):
            approximator.fit(dataset=train_dataset, validation_data=validation_dataset, epochs=num_epochs)

        output = stream.getvalue()

    print(output)
    if isinstance(approximator, EnsembleApproximator):
        print(keras.tree.map_structure(keras.ops.shape, train_dataset[0]))

    # check that there is a progress bar
    assert "━" in output, "no progress bar"

    # check that the loss is shown
    assert "loss" in output
    assert re.search(r"\bloss: \d+\.\d+", output) is not None, "training loss not correctly shown"

    # check that validation loss is shown
    assert "val_loss" in output
    assert re.search(r"\bval_loss: \d+\.\d+", output) is not None, "validation loss not correctly shown"

    # check that the shown loss is not nan or zero
    assert re.search(r"\bnan\b", output) is None, "found nan in output"
    assert re.search(r"\bloss: 0\.0000e\+00\b", output) is None, "found zero loss in output"


def test_loss_progress_online(approximator, simulator, adapter, validation_dataset):
    check_combination_simulator_adapter(simulator, adapter)

    approximator.compile(optimizer="AdamW")
    num_epochs = 3

    train_dataset = OnlineDataset(simulator=simulator, adapter=adapter, num_batches=4, batch_size=16)
    if isinstance(approximator, EnsembleApproximator):
        train_dataset = EnsembleDataset(train_dataset, member_names=list(approximator.approximators.keys()))

    # Capture ostream and train model
    with io.StringIO() as stream:
        with redirect_stdout(stream):
            approximator.fit(
                dataset=train_dataset,
                validation_data=validation_dataset,
                epochs=num_epochs,
            )

        output = stream.getvalue()

    print(output)

    # check that there is a progress bar
    assert "━" in output, "no progress bar"

    # check that the loss is shown
    assert "loss" in output
    assert re.search(r"\bloss: \d+\.\d+", output) is not None, "training loss not correctly shown"

    # check that validation loss is shown
    assert "val_loss" in output
    assert re.search(r"\bval_loss: \d+\.\d+", output) is not None, "validation loss not correctly shown"

    # check that the shown loss is not nan or zero
    assert re.search(r"\bnan\b", output) is None, "found nan in output"
    assert re.search(r"\bloss: 0\.0000e\+00\b", output) is None, "found zero loss in output"
