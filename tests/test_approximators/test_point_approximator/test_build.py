from tests.utils.approximator_checks import check_build


def test_build(point_approximator, simulator, batch_size, adapter):
    check_build(point_approximator, simulator, batch_size, adapter)
