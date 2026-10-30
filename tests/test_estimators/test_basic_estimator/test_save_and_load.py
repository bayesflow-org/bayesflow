from tests.utils.estimator_checks import check_save_and_load


def test_save_and_load(tmp_path, basic_estimator, train_dataset, validation_dataset):
    check_save_and_load(tmp_path, basic_estimator, train_dataset)
