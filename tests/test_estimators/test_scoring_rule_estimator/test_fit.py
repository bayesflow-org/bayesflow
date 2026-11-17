from tests.utils.estimator_checks import check_fit


def test_loss_progress(scoring_rule_estimator_any, train_dataset, validation_dataset):
    check_fit(scoring_rule_estimator_any, train_dataset, validation_dataset)
