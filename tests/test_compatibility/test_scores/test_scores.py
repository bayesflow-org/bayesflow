import pytest
from utils import SaveLoadTest, save_config, load_from_config, dump_path, load_path
import numpy as np
import keras


@pytest.mark.parametrize(
    "scoring_rule",
    [
        ["median_score", {}],
        ["mean_score", {}],
        ["normed_diff_score", dict(k=3)],
        ["quantile_score", {}],
        ["multivariate_normal_score", {}],
    ],
    indirect=True,
)
class TestScore(SaveLoadTest):
    filenames = {
        "model": "model.pickle",
        "output": "output.pickle",
    }

    @pytest.fixture
    def setup(self, filepaths, mode, scoring_rule, random_samples, request):
        if mode == "save":
            save_config(scoring_rule, filepaths["model"])

            output = self.evaluate(scoring_rule, random_samples)
            dump_path(output, filepaths["output"])

        scoring_rule = load_from_config(filepaths["model"])
        output = load_path(filepaths["output"])

        return scoring_rule, output

    def evaluate(self, scoring_rule, data):
        # Using random data also as targets for the purpose of this test.
        head_shapes = scoring_rule.get_head_shapes_from_target_shape(data.shape)
        estimates = {}
        for key, output_shape in head_shapes.items():
            link = scoring_rule.get_link(key)
            if hasattr(link, "compute_input_shape"):
                link_input_shape = link.compute_input_shape(output_shape)
            else:
                link_input_shape = output_shape
            dummy_input = keras.ops.ones((data.shape[0],) + link_input_shape)
            estimates[key] = link(dummy_input)

        score = scoring_rule.score(estimates, data)
        return score

    def test_output(self, setup, random_samples):
        scoring_rule, reference = setup
        output = self.evaluate(scoring_rule, random_samples)
        np.testing.assert_allclose(reference, output)
