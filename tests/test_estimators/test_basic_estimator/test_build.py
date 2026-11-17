from tests.utils.estimator_checks import check_build


def test_build(basic_estimator, simulator, batch_size, adapter):
    check_build(basic_estimator, simulator, batch_size, adapter)
