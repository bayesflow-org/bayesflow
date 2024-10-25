import keras
import numpy as np


def test_two_moons(simulator, batch_size):
    samples = simulator.sample((batch_size,))

    assert isinstance(samples, dict)
    assert list(samples.keys()) == ["r", "alpha", "theta", "x"]
    assert all(isinstance(value, np.ndarray) for value in samples.values())

    assert samples["r"].shape == (batch_size, 1)
    assert samples["alpha"].shape == (batch_size, 1)
    assert samples["theta"].shape == (batch_size, 2)
    assert samples["x"].shape == (batch_size, 2)


def test_sample(simulator, batch_size):
    samples = simulator.sample((batch_size,))

    # test output structure
    assert isinstance(samples, dict)

    for key, value in samples.items():
        print(f"{key}.shape = {keras.ops.shape(value)}")

        # test type
        assert isinstance(value, np.ndarray)

        # test shape
        assert value.shape[0] == batch_size

        # test batch randomness
        assert not np.allclose(value, value[0])
