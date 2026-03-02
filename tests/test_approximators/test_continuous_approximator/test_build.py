from tests.utils.approximator_checks import check_build


def test_build(continuous_approximator, simulator, batch_size, adapter):
    check_build(continuous_approximator, simulator, batch_size, adapter)
