import pytest
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


def test_ensemble_batch_shape_and_type(ensemble_dataset, ensemble_size, batch_size):
    batch = ensemble_dataset[0]
    assert "x" in batch
    x = batch["x"]

    assert isinstance(x, np.ndarray)
    assert x.ndim == 3
    assert x.shape[1] == ensemble_size
    assert x.shape[0] <= batch_size
    assert x.shape[2] == 2


def test_data_reuse_one_means_identical_members(ensemble_dataset, data_reuse):
    if data_reuse != 1.0:
        pytest.skip("Only checks the data_reuse=1 case.")
    x = ensemble_dataset[0]["x"]
    # member 0 and 1 identical
    assert np.allclose(x[:, 0, :], x[:, 1, :])


def test_data_reuse_zero_means_not_identical_members(ensemble_dataset, data_reuse):
    if data_reuse != 0.0:
        pytest.skip("Only checks the data_reuse=0 case.")
    x = ensemble_dataset[0]["x"]
    assert not np.allclose(x[:, 0, :], x[:, 1, :])


def overlap(a, b):
    return len(set(a.tolist()).intersection(b.tolist())) / len(a)


def test_offline_overlap_monotonic(offline_dataset, ensemble_size):
    from bayesflow import EnsembleDataset

    ds0 = EnsembleDataset(offline_dataset, ensemble_size=ensemble_size, data_reuse=0.0)
    ds05 = EnsembleDataset(offline_dataset, ensemble_size=ensemble_size, data_reuse=0.5)
    ds1 = EnsembleDataset(offline_dataset, ensemble_size=ensemble_size, data_reuse=1.0)

    # assuming indexed impl
    a0, b0 = ds0._wrapped.member_indices[0], ds0._wrapped.member_indices[1]
    a05, b05 = ds05._wrapped.member_indices[0], ds05._wrapped.member_indices[1]
    a1, b1 = ds1._wrapped.member_indices[0], ds1._wrapped.member_indices[1]

    assert overlap(a0, b0) <= overlap(a05, b05) <= overlap(a1, b1)
