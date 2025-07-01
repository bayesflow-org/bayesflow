import pytest
from utils import SaveLoadTest, load_from_config, save_config, load_path, dump_path
import numpy as np


class TestAdapter(SaveLoadTest):
    filenames = {
        "model": "model.pickle",
        "output": "output.pickle",
    }

    @pytest.fixture
    def setup(self, filepaths, mode, adapter, data_1, data_2):
        if mode == "save":
            _ = adapter(data_1)
            save_config(adapter, filepaths["model"])

            output = self.evaluate(adapter, data_2)
            dump_path(output, filepaths["output"])

        adapter = load_from_config(filepaths["model"])
        output = load_path(filepaths["output"])

        return adapter, output

    def evaluate(self, adapter, data):
        adapted = adapter(data)
        cycled = adapter(adapted, inverse=True)
        return {"adapted": adapted, "cycled": cycled}

    def test_output(self, setup, data_2):
        adapter, reference = setup
        output = self.evaluate(adapter, data_2)
        for k, v in reference.items():
            for name, variable in v.items():
                if name == "s3":
                    continue
                np.testing.assert_allclose(
                    variable, output[k][name], err_msg=f"Values for key '{k}/{name} do not match."
                )
