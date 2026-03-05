from tests.utils.approximator_checks import check_save_and_load


def test_save_and_load(tmp_path, scoring_rule_approximator_any, train_dataset, validation_dataset):
    check_save_and_load(tmp_path, scoring_rule_approximator_any, train_dataset)
