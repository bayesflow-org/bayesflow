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
        ["fusion_network", "fusion", "fusion_adapter"],
    ],
    indirect=True,
)
class TestWorkflow(SaveLoadTest):
    filenames = {
        "approximator": "approximator.keras",
        "input": "input.pickle",
        "output": "output.pickle",
    }

    @pytest.fixture()
    def setup(self, filepaths, mode, inference_network, summary_network, simulator, adapter):
        if mode == "save":
            import bayesflow as bf

            workflow = bf.BasicWorkflow(
                adapter=adapter,
                inference_network=inference_network,
                summary_network=summary_network,
                inference_variables=["parameters"],
                summary_variables=["observables"],
                simulator=simulator,
            )
            workflow.fit_online(epochs=1, batch_size=8, num_batches_per_epoch=2, verbose=0)
            keras.saving.save_model(workflow.approximator, filepaths["approximator"])

            input = workflow.simulate(4)
            output = self.evaluate(workflow.approximator, input)
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
        print(reference)
        np.testing.assert_allclose(reference, output)
