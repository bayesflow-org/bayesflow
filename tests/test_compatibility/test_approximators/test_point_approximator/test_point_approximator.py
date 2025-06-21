import pytest
from utils import SaveLoadTest, dump_path, load_path
import numpy as np
import keras


@pytest.mark.parametrize(
    "summary_network,simulator,standardize",
    [
        [None, "normal", "all"],
    ],
    indirect=True,
)
class TestPointApproximator(SaveLoadTest):
    filenames = {
        "approximator": "approximator.keras",
        "input": "input.pickle",
        "output": "output.pickle",
    }

    @pytest.fixture()
    def setup(
        self, filepaths, mode, approximator, adapter, point_inference_network, summary_network, standardize, simulator
    ):
        if mode == "save":
            approximator.compile("adamw", run_eagerly=False)
            approximator.fit(simulator=simulator, epochs=1, batch_size=8, num_batches=2, verbose=0)
            keras.saving.save_model(approximator, filepaths["approximator"])

            input = simulator.sample(4)
            output = self.evaluate(approximator, input)
            dump_path(input, filepaths["input"])
            dump_path(output, filepaths["output"])

        approximator = keras.saving.load_model(filepaths["approximator"])
        input = load_path(filepaths["input"])
        output = load_path(filepaths["output"])

        return approximator, input, output

    def evaluate(self, approximator, data):
        return approximator.estimate(data)

    def test_output(self, setup):
        approximator, input, reference = setup
        output = self.evaluate(approximator, input)

        from keras.tree import flatten

        for ref, out in zip(flatten(reference), flatten(output)):
            np.testing.assert_allclose(ref, out)
