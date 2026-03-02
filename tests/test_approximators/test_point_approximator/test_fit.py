from tests.utils.approximator_checks import check_fit


def test_loss_progress(point_approximator, train_dataset, validation_dataset):
    check_fit(point_approximator, train_dataset, validation_dataset)
