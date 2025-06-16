from utils import SaveLoadTest, load_from_config, save_config
import numpy as np
import keras
import pytest


class TestDistribution(SaveLoadTest):
    filenames = {
        "model": "model.pickle",
        "output": "output.npy",
    }

    @pytest.fixture
    def setup(self, filepaths, mode, distribution, random_samples):
        if mode == "save":
            distribution.build(keras.ops.shape(random_samples))
            save_config(distribution, filepaths["model"])

            output = self.evaluate(distribution, random_samples)
            np.save(filepaths["output"], output, allow_pickle=False)

        distribution = load_from_config(filepaths["model"])
        output = np.load(filepaths["output"])

        return distribution, output

    def evaluate(self, distribution, random_samples):
        return keras.ops.convert_to_numpy(distribution.log_prob(random_samples))

    def test_output(self, setup, random_samples):
        distribution, output = setup
        np.testing.assert_allclose(self.evaluate(distribution, random_samples), output)
