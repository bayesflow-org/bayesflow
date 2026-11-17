from tests.utils.estimator_checks import check_save_and_load


def test_save_and_load(tmp_path, scoring_rule_estimator_any, train_dataset, validation_dataset):
    check_save_and_load(tmp_path, scoring_rule_estimator_any, train_dataset)
