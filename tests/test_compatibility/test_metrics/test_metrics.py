import pytest
from utils import SaveLoadTest, load_from_config, save_config
import numpy as np
import keras


@pytest.mark.parametrize(
    "metric",
    [
        ["root_mean_squared_error", dict(normalize=True, dtype="float32")],
        ["root_mean_squared_error", dict(normalize=False)],
        ["maximum_mean_discrepancy", dict(kernel="gaussian", unbiased=True, dtype="float32")],
        ["maximum_mean_discrepancy", dict(kernel="inverse_multiquadratic", unbiased=False)],
    ],
    indirect=True,
)
class TestMetric(SaveLoadTest):
    filenames = {
        "model": "model.pickle",
        "output": "output.npy",
    }

    @pytest.fixture
    def setup(self, filepaths, mode, metric, samples_1, samples_2):
        if mode == "save":
            save_config(metric, filepaths["model"])

            output = self.evaluate(metric, samples_1, samples_2)
            np.save(filepaths["output"], output, allow_pickle=False)

        metric = load_from_config(filepaths["model"])
        output = np.load(filepaths["output"])

        return metric, output

    def evaluate(self, metric, samples_1, samples_2):
        return keras.ops.convert_to_numpy(metric(samples_1, samples_2))

    def test_output(self, setup, samples_1, samples_2):
        metric, output = setup
        np.testing.assert_allclose(self.evaluate(metric, samples_1, samples_2), output)
