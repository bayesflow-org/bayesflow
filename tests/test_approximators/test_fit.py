import re

import pytest
import io
from contextlib import redirect_stdout


@pytest.mark.skip(reason="not implemented")
def test_compile(approximator):
    approximator.compile(optimizer="AdamW")


@pytest.mark.skip(reason="not implemented")
def test_fit(approximator, dataset):
    approximator.compile(optimizer="AdamW")
    approximator.fit(dataset)

    assert approximator.losses is not None


def test_loss_progress(approximator, train_dataset, validation_dataset):
    approximator.compile(optimizer="AdamW")
    num_epochs = 2

    # Capture ostream and train model
    with io.StringIO() as stream:
        with redirect_stdout(stream):
            approximator.fit(dataset=train_dataset, validation_data=validation_dataset, epochs=num_epochs)

        output = stream.getvalue()

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
