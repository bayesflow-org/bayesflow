import pytest
from utils import SaveLoadTest, dump_path, load_path
import numpy as np
import keras


@pytest.mark.parametrize(
    "summary_network",
    [
        ["time_series_network", dict()],
        ["time_series_transformer", dict()],
        ["fusion_transformer", dict()],
        ["set_transformer", dict()],
        ["deep_set", dict()],
    ],
    indirect=True,
)
class TestSummaryNetwork(SaveLoadTest):
    filenames = {
        "model": "model.keras",
        "output": "output.pickle",
    }

    @pytest.fixture()
    def setup(self, filepaths, mode, summary_network, summary_dim, random_set):
        if mode == "save":
            shape = keras.ops.shape(random_set)
            summary_network.build(shape)

            _ = summary_network(random_set)
            keras.saving.save_model(summary_network, filepaths["model"])
            output = self.evaluate(summary_network, random_set)

            dump_path(output, filepaths["output"])

        summary_network = keras.saving.load_model(filepaths["model"])
        output = load_path(filepaths["output"])

        return summary_network, random_set, output

    def evaluate(self, summary_network, data):
        return keras.ops.convert_to_numpy(summary_network(data))

    def test_output(self, setup):
        approximator, data, reference = setup
        output = self.evaluate(approximator, data)

        np.testing.assert_allclose(reference, output)
