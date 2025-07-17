import re

import io
from contextlib import redirect_stdout


def test_loss_progress(continuous_approximator_ensemble, train_dataset_for_ensemble, validation_dataset):
    continuous_approximator_ensemble.compile(optimizer="AdamW")
    num_epochs = 3

    # Capture ostream and train model
    with io.StringIO() as stream:
        with redirect_stdout(stream):
            continuous_approximator_ensemble.fit(
                dataset=train_dataset_for_ensemble, validation_data=validation_dataset, epochs=num_epochs
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
