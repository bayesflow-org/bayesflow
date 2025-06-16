import pytest
from utils import SaveLoadTest, dump_path, load_path
import numpy as np
import keras


@pytest.mark.parametrize("inference_network", ["coupling_flow", "flow_matching"], indirect=True)
@pytest.mark.parametrize(
    "summary_network,simulator,adapter",
    [
        ["time_series_transformer", "sir", None],
        ["fusion_transformer", "sir", None],
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
    def setup(self, filepaths, mode, inference_network, summary_network, simulator, adapter):
        if mode == "save":
            import bayesflow as bf

            approximator = bf.approximators.ContinuousApproximator(
                adapter=adapter,
                inference_network=inference_network,
                summary_network=summary_network,
            )
            approximator.compile("adamw")
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
