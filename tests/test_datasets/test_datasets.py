import numpy as np
import keras
import pickle


def test_dataset_is_picklable(any_dataset):
    pickled = pickle.loads(pickle.dumps(any_dataset))

    assert type(pickled) is type(any_dataset)

    samples = any_dataset[0]  # dict of {param_name: param_value}
    pickled_samples = pickled[0]
    assert isinstance(samples, dict)
    assert isinstance(pickled_samples, dict)

    assert list(samples.keys()) == list(pickled_samples.keys())

    for key in samples.keys():
        assert keras.ops.shape(samples[key]) == keras.ops.shape(pickled_samples[key])


def test_dataset_works_in_fit(model, individual_dataset):
    print(next(iter(individual_dataset[0].values())).dtype)
    model.fit(individual_dataset, epochs=1, steps_per_epoch=1)


def test_dataset_returns_batch(any_dataset, batch_size):
    samples = any_dataset[0]  # dict of {param_name: param_value}
    samples = next(iter(samples.values()))  # first param value

    assert keras.ops.shape(samples)[0] == batch_size


def test_ensemble_batch_shape_and_type(ensemble_dataset, num_ensemble, batch_size):
    batch = ensemble_dataset[0]
    assert "x" in batch
    x = batch["x"]

    assert isinstance(x, np.ndarray)
    assert x.ndim == 3
    assert x.shape[1] == num_ensemble
    assert x.shape[0] <= batch_size
    assert x.shape[2] == 2
