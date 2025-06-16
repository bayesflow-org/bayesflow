import pytest
from utils import save_config, load_from_config, dump_path, load_path
from utils import SaveLoadTest
import numpy as np


class TestLink(SaveLoadTest):
    filenames = {
        "model": "model.pickle",
        "output": "output.pickle",
    }

    @pytest.fixture
    def setup(self, filepaths, mode, link, random_samples):
        if mode == "save":
            _ = link(random_samples)
            save_config(link, filepaths["model"])

            output = self.evaluate(link, random_samples)
            dump_path(output, filepaths["output"])

        link = load_from_config(filepaths["model"])
        output = load_path(filepaths["output"])

        return link, output

    def evaluate(self, link, data):
        return link(data)

    def test_output(self, setup, random_samples):
        link, reference = setup
        print(reference)
        output = self.evaluate(link, random_samples)
        np.testing.assert_allclose(reference, output)
