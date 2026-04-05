from tests.utils.approximator_checks import check_build


def test_build(scoring_rule_approximator_any, simulator, batch_size, adapter):
    check_build(scoring_rule_approximator_any, simulator, batch_size, adapter)
