import pytest
from utils import SaveLoadTest, dump_path, load_path
import numpy as np
import keras


@pytest.mark.parametrize(
    "inference_network",
    [
        [
            "coupling_flow",
            dict(
                depth=2,
                subnet="mlp",
                subnet_kwargs=dict(widths=[8, 8]),
                transform="affine",
                transform_kwargs=dict(clamp=1.8),
            ),
        ],
        [
            "coupling_flow",
            dict(
                depth=2,
                subnet="mlp",
                subnet_kwargs=dict(widths=[8, 8]),
                transform="spline",
                transform_kwargs=dict(bins=8),
            ),
        ],
        ["flow_matching", dict(integrate_kwargs={"method": "rk45", "steps": 10})],
        ["consistency_model", dict(total_steps=10)],
        [
            "diffusion_model",
            dict(noise_schedule="edm", prediction_type="F", integrate_kwargs={"method": "rk45", "steps": 10}),
        ],
        [
            "diffusion_model",
            dict(noise_schedule="edm", prediction_type="velocity", integrate_kwargs={"method": "euler", "steps": 10}),
        ],
        [
            "diffusion_model",
            dict(noise_schedule="edm", prediction_type="noise", integrate_kwargs={"method": "euler", "steps": 10}),
        ],
        [
            "diffusion_model",
            dict(noise_schedule="cosine", prediction_type="F", integrate_kwargs={"method": "euler", "steps": 10}),
        ],
        [
            "diffusion_model",
            dict(
                noise_schedule="cosine", prediction_type="velocity", integrate_kwargs={"method": "euler", "steps": 10}
            ),
        ],
        [
            "free_form_flow",
            dict(encoder_subnet_kwargs={"widths": [16, 16]}, decoder_subnet_kwargs={"widths": [16, 16]}),
        ],
        ["point_inference_network", dict(subnet_kwargs={"widths": [8, 8]})],
    ],
    indirect=True,
)
class TestInferenceNetwork(SaveLoadTest):
    filenames = {
        "model": "model.keras",
        "output": "output.pickle",
    }

    @pytest.fixture()
    def setup(self, filepaths, mode, inference_network, random_samples, random_conditions):
        if mode == "save":
            xz_shape = keras.ops.shape(random_samples)
            conditions_shape = keras.ops.shape(random_conditions) if random_conditions is not None else None
            inference_network.build(xz_shape, conditions_shape)

            _ = inference_network.compute_metrics(random_samples, conditions=random_conditions)
            keras.saving.save_model(inference_network, filepaths["model"])
            output = self.evaluate(inference_network, random_samples, random_conditions)

            dump_path(output, filepaths["output"])

        inference_network = keras.saving.load_model(filepaths["model"])
        output = load_path(filepaths["output"])

        return inference_network, random_samples, random_conditions, output

    def evaluate(self, inference_network, samples, conditions):
        import bayesflow as bf

        if isinstance(inference_network, bf.networks.ConsistencyModel):
            # not invertible, but inverse with steps=1 is deterministic
            return keras.tree.map_structure(
                keras.ops.convert_to_numpy, inference_network._inverse(samples, conditions, steps=1)
            )
        if isinstance(inference_network, bf.networks.PointInferenceNetwork) and conditions is None:
            pytest.skip("PointInferenceNetwork requires condition")
        try:
            return keras.tree.map_structure(
                keras.ops.convert_to_numpy, inference_network.log_prob(samples, conditions=conditions)
            )
        except NotImplementedError:
            pytest.skip("log_prob not available")

    def test_output(self, setup):
        approximator, samples, conditions, reference = setup
        output = self.evaluate(approximator, samples, conditions)
        print(reference)
        from keras.tree import flatten

        for ref, out in zip(flatten(reference), flatten(output)):
            print(ref, out)
            np.testing.assert_allclose(ref, out)
