import numpy as np


def test_multi_dataset(dataset, labeled_dataset, unlabeled_dataset):
    assert dataset.num_batches == labeled_dataset.num_batches

    # test returned keys
    batch = dataset[0]
    assert set(batch.keys()) == {"labeled", "unlabeled"}
    assert set(batch["labeled"].keys()) == {"y", "mu", "tau"}
    assert set(batch["unlabeled"].keys()) == {"y"}

    # test wrapping
    batch_wrapped = dataset[2]
    assert all(np.allclose(batch["unlabeled"][key], batch_wrapped["unlabeled"][key]) for key in batch["unlabeled"])

    assert set(dataset.get_config()["datasets"].keys()) == {"labeled", "unlabeled"}
