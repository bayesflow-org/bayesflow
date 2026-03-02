from tests.utils.approximator_checks import check_save_and_load


def test_save_and_load(tmp_path, continuous_approximator, train_dataset, validation_dataset):
    check_save_and_load(tmp_path, continuous_approximator, train_dataset)
