from tests.utils.approximator_checks import check_fit


def test_loss_progress(continuous_approximator, train_dataset, validation_dataset):
    check_fit(continuous_approximator, train_dataset, validation_dataset)
