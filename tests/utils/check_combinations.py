import pytest


def check_combination_simulator_adapter(simulator, adapter):
    """Make sure simulator and adapter fixtures fit together and appropriate errors are raised if not."""
    # check whether the simulator returns a 'weight' key
    simulator_with_sample_weight = "weight" in simulator.sample(1).keys()
    # scan adapter representation for occurance of a rename pattern for 'sample_weight'
    adapter_with_sample_weight = "-> 'sample_weight'" in str(adapter)

    if not simulator_with_sample_weight and adapter_with_sample_weight:
        # adapter should expect a 'weight' key and raise a KeyError.
        with pytest.raises(KeyError):
            adapter(simulator.sample(1))
        # Don't use this fixture combination for further tests.
        pytest.skip(reason="Do not use this fixture combination for further tests")  # TODO: better reason
    elif simulator_with_sample_weight and not adapter_with_sample_weight:
        # When a weight key is present, but the adapter does not configure it
        # to be used as sample weight, no error is raised currently.
        # Don't use this fixture combination for further tests.
        pytest.skip(reason="Do not use this fixture combination for further tests")  # TODO: better reason


def check_approximator_multivariate_normal_score(approximator):
    from bayesflow.approximators import PointApproximator
    from bayesflow.scores import MultivariateNormalScore

    if isinstance(approximator, PointApproximator):
        for score in approximator.inference_network.scores.values():
            if isinstance(score, MultivariateNormalScore):
                pytest.skip(reason="MultivariateNormalScore is unstable")
