import io
import re
from contextlib import redirect_stdout

import keras

from tests.utils.assertions import assert_models_equal
from tests.utils.check_combinations import check_combination_simulator_adapter


def check_build(approximator, simulator, batch_size, adapter):
    """Verify that an approximator builds correctly from simulated data."""
    check_combination_simulator_adapter(simulator, adapter)

    num_batches = 4
    data = simulator.sample((num_batches * batch_size,))

    batch = adapter(data)
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, batch)
    batch_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    approximator.build(batch_shapes)
    for layer in approximator.standardizer.standardize_layers.values():
        assert layer.built
        for count in layer.count:
            assert count == 0.0


def check_fit(approximator, train_dataset, validation_dataset):
    """Verify that an approximator trains and reports loss correctly."""
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


def check_save_and_load(tmp_path, approximator, train_dataset):
    """Verify that an approximator can be saved and loaded."""
    # to save, the model must be built
    batch = keras.tree.map_structure(keras.ops.convert_to_tensor, train_dataset[0])
    data_shapes = keras.tree.map_structure(keras.ops.shape, batch)
    approximator.build(data_shapes)
    approximator.compute_metrics(**batch)

    keras.saving.save_model(approximator, tmp_path / "model.keras")
    loaded = keras.saving.load_model(tmp_path / "model.keras")

    assert_models_equal(approximator, loaded)
