from tests.utils.estimator_checks import check_fit


def test_loss_progress(basic_estimator, train_dataset, validation_dataset):
    check_fit(basic_estimator, train_dataset, validation_dataset)
