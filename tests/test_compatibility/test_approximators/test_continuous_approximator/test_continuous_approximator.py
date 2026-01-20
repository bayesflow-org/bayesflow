import pytest
from utils import SaveLoadTest, dump_path, load_path
import numpy as np
import keras


@pytest.mark.parametrize("inference_network", ["coupling_flow"], indirect=True)
@pytest.mark.parametrize(
    "summary_network,simulator,adapter,standardize",
    [
        ["deep_set", "sir", "summary", ["summary_variables", "inference_variables"]],  # use deep_set for speed
        [None, "two_moons", "direct", "all"],
        [None, "two_moons", "direct", None],
    ],
    indirect=True,
)
class TestContinuousApproximator(SaveLoadTest):
    filenames = {
        "approximator": "approximator.keras",
        "input": "input.pickle",
        "output": "output.pickle",
    }

    @pytest.fixture()
    def setup(self, filepaths, mode, approximator, adapter, inference_network, summary_network, standardize, simulator):
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
        return approximator.log_prob(data)

    def test_output(self, setup):
        approximator, input, reference = setup
        output = self.evaluate(approximator, input)
        np.testing.assert_allclose(reference, output)
