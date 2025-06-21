from utils import SaveLoadTest
import numpy as np
import keras
import pytest


@pytest.mark.parametrize(
    "distribution",
    [
        ["diagonal_normal", dict(trainable_parameters=False)],
        ["diagonal_normal", dict(trainable_parameters=True)],
        ["diagonal_student_t", dict(trainable_parameters=False)],
        ["diagonal_student_t", dict(trainable_parameters=True)],
        ["mixture", dict(trainable_mixture=False)],
        ["mixture", dict(trainable_mixture=True)],
    ],
    indirect=True,
)
class TestDistribution(SaveLoadTest):
    filenames = {
        "model": "model.keras",
        "output": "output.npy",
    }

    @pytest.fixture
    def setup(self, filepaths, mode, distribution, random_samples):
        from bayesflow.utils.serialization import serialize, deserialize

        class DummyModel(keras.Model):
            def __init__(self, distribution, **kwargs):
                super().__init__(**kwargs)
                self.distribution = distribution

            def call(self, inputs):
                return self.distribution.log_prob(inputs)

            def get_config(self):
                base_config = super().get_config()
                config = {"distribution": self.distribution}
                return base_config | serialize(config)

            @classmethod
            def from_config(cls, config, custom_objects=None):
                return cls(**deserialize(config, custom_objects=custom_objects))

        if mode == "save":
            distribution.build(keras.ops.shape(random_samples))

            model = DummyModel(distribution)
            model.compile(loss=keras.losses.MeanSquaredError())
            model.fit(
                random_samples,
                keras.ops.ones(keras.ops.shape(random_samples)[:-1]),
                batch_size=keras.ops.shape(random_samples)[0],
                epochs=1,
            )
            model.save(filepaths["model"])

            output = self.evaluate(model.distribution, random_samples)
            np.save(filepaths["output"], output, allow_pickle=False)

        distribution = keras.saving.load_model(
            filepaths["model"], custom_objects={"DummyModel": DummyModel}
        ).distribution
        output = np.load(filepaths["output"])

        return distribution, output

    def evaluate(self, distribution, random_samples):
        return keras.ops.convert_to_numpy(distribution.log_prob(random_samples))

    def test_output(self, setup, random_samples):
        distribution, output = setup
        np.testing.assert_allclose(self.evaluate(distribution, random_samples), output)
