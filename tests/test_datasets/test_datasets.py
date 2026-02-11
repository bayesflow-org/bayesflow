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

    samples_shape = keras.tree.map_structure(keras.ops.shape, samples)
    pickled_samples_shape = keras.tree.map_structure(keras.ops.shape, pickled_samples)
    assert samples_shape == pickled_samples_shape


def test_dataset_works_in_fit(model, individual_dataset):
    print(next(iter(individual_dataset[0].values())).dtype)
    model.fit(individual_dataset, epochs=1, steps_per_epoch=1)


def test_dataset_returns_batch(individual_dataset, batch_size):
    samples = individual_dataset[0]  # dict of {param_name: param_value}
    samples = next(iter(samples.values()))  # first param value

    assert keras.ops.shape(samples)[0] == batch_size


def test_dataset_returns_batch_ensemble(ensemble_dataset, batch_size):
    samples = ensemble_dataset[0]  # dict of {param_name: {member_name: param_value_for_member}}
    samples = next(iter(samples.values()))  # first param
    samples = next(iter(samples.values()))  # first member

    assert keras.ops.shape(samples)[0] == batch_size


def test_ensemble_batch_shape_and_type(ensemble_dataset, member_names, batch_size):
    batch = ensemble_dataset[0]
    assert "x" in batch
    x = batch["x"]

    assert isinstance(x, dict)
    assert set(x.keys()) == set(member_names)

    x_individual = x[member_names[0]]
    assert x_individual.shape[0] == batch_size


def test_data_reuse_one_means_identical_members(ensemble_dataset, data_reuse, member_names):
    if data_reuse != 1.0:
        pytest.skip("Only checks the data_reuse=1 case.")
    x = ensemble_dataset[0]["x"]
    m1, m2 = member_names[:2]
    assert np.allclose(x[m1], x[m2])


def test_data_reuse_zero_means_not_identical_members(ensemble_dataset, data_reuse, member_names):
    if data_reuse != 0.0:
        pytest.skip("Only checks the data_reuse=0 case.")
    x = ensemble_dataset[0]["x"]
    m1, m2 = member_names[:2]
    assert not np.allclose(x[m1], x[m2])


def overlap(a, b):
    return len(set(a.tolist()).intersection(b.tolist())) / len(a)


def test_offline_overlap_monotonic(offline_dataset, member_names):
    from bayesflow import EnsembleDataset

    ds0 = EnsembleDataset(offline_dataset, member_names=member_names, data_reuse=0.0)
    ds05 = EnsembleDataset(offline_dataset, member_names=member_names, data_reuse=0.5)
    ds1 = EnsembleDataset(offline_dataset, member_names=member_names, data_reuse=1.0)

    # assuming wrapped dataset is indexed
    m1, m2 = member_names[:2]
    a0, b0 = ds0._wrapped.member_indices[m1], ds0._wrapped.member_indices[m2]
    a05, b05 = ds05._wrapped.member_indices[m1], ds05._wrapped.member_indices[m2]
    a1, b1 = ds1._wrapped.member_indices[m1], ds1._wrapped.member_indices[m2]

    assert overlap(a0, b0) <= overlap(a05, b05) <= overlap(a1, b1)

    assert len(ds0) < len(ds1)


def test_ensemble_value_error(individual_dataset):
    from bayesflow import EnsembleDataset

    with pytest.raises(ValueError):
        EnsembleDataset(individual_dataset, member_names=["just_one"])

    for data_reuse in [-0.5, 2]:
        with pytest.raises(ValueError):
            EnsembleDataset(individual_dataset, member_names=["a", "b"], data_reuse=data_reuse)

    ensemble_dataset = EnsembleDataset(individual_dataset, member_names=["a", "b"])
    with pytest.raises(TypeError):
        EnsembleDataset(ensemble_dataset, member_names=["a", "b"])

    dummy_object = "abc"
    with pytest.raises(TypeError):
        EnsembleDataset(dummy_object, member_names=["a", "b"])  # type: ignore
