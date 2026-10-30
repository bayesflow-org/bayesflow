import keras
import pytest

import bayesflow as bf
from bayesflow import approximators, estimators


def _base(alias):
    return alias()


def _basic(alias):
    return alias(inference_network=bf.networks.CouplingFlow())


def _autoregressive(alias):
    return alias(inference_network=bf.networks.CouplingFlow())


def _compositional(alias):
    return alias(inference_network=bf.networks.CouplingFlow())


def _scoring_rule(alias):
    return alias(inference_network=bf.networks.PointNetwork(points="mean"))


def _model_comparison(alias):
    network = bf.networks.ScoringRuleNetwork(
        scoring_rules={"cross_entropy": bf.scoring_rules.CrossEntropyScore()},
        subnet_kwargs={"widths": (4,)},
    )
    return alias(inference_network=network)


def _ratio(alias):
    return alias(inference_network=keras.layers.Dense(4))


def _ensemble(alias):
    instance = alias(approximators={})
    assert instance.approximators is instance.estimators
    return instance


@pytest.mark.parametrize(
    ("old_name", "new_name", "factory"),
    [
        ("Approximator", "Estimator", _base),
        ("ContinuousApproximator", "BasicEstimator", _basic),
        ("AutoregressiveApproximator", "AutoregressiveEstimator", _autoregressive),
        ("CompositionalApproximator", "CompositionalEstimator", _compositional),
        ("ScoringRuleApproximator", "ScoringRuleEstimator", _scoring_rule),
        ("ModelComparisonApproximator", "ModelComparisonEstimator", _model_comparison),
        ("RatioApproximator", "RatioEstimator", _ratio),
        ("EnsembleApproximator", "EnsembleEstimator", _ensemble),
    ],
)
def test_deprecated_alias(old_name, new_name, factory):
    alias = getattr(approximators, old_name)
    replacement = getattr(estimators, new_name)

    assert alias.__bases__ == (replacement,)
    with pytest.warns(DeprecationWarning, match=rf"{old_name}.*{new_name}"):
        instance = factory(alias)
    assert isinstance(instance, replacement)


@pytest.mark.parametrize(
    ("old_name", "new_name"),
    [
        ("ContinuousApproximator", "BasicEstimator"),
        ("AutoregressiveApproximator", "AutoregressiveEstimator"),
        ("CompositionalApproximator", "CompositionalEstimator"),
        ("ScoringRuleApproximator", "ScoringRuleEstimator"),
        ("ModelComparisonApproximator", "ModelComparisonEstimator"),
        ("RatioApproximator", "RatioEstimator"),
        ("EnsembleApproximator", "EnsembleEstimator"),
    ],
)
def test_top_level_alias(old_name, new_name):
    assert getattr(bf, old_name) is getattr(approximators, old_name)
    assert getattr(bf, new_name) is getattr(estimators, new_name)


def test_backend_approximator_alias():
    from bayesflow.approximators.backend_approximators import BackendApproximator
    from bayesflow.estimators.backend_estimators import BackendEstimator

    assert BackendApproximator.__bases__ == (BackendEstimator,)
    with pytest.warns(DeprecationWarning, match="BackendApproximator.*BackendEstimator"):
        instance = BackendApproximator()
    assert isinstance(instance, BackendEstimator)


@pytest.mark.jax
def test_jax_approximator_alias():
    from bayesflow.approximators.backend_approximators.jax_approximator import JAXApproximator
    from bayesflow.estimators.backend_estimators.jax_estimator import JAXEstimator

    assert JAXApproximator.__bases__ == (JAXEstimator,)
    with pytest.warns(DeprecationWarning, match="JAXApproximator.*JAXEstimator"):
        instance = JAXApproximator()
    assert isinstance(instance, JAXEstimator)


@pytest.mark.tensorflow
def test_tensorflow_approximator_alias():
    from bayesflow.approximators.backend_approximators.tensorflow_approximator import TensorFlowApproximator
    from bayesflow.estimators.backend_estimators.tensorflow_estimator import TensorFlowEstimator

    assert TensorFlowApproximator.__bases__ == (TensorFlowEstimator,)
    with pytest.warns(DeprecationWarning, match="TensorFlowApproximator.*TensorFlowEstimator"):
        instance = TensorFlowApproximator()
    assert isinstance(instance, TensorFlowEstimator)


@pytest.mark.torch
def test_torch_approximator_alias():
    from bayesflow.approximators.backend_approximators.torch_approximator import TorchApproximator
    from bayesflow.estimators.backend_estimators.torch_estimator import TorchEstimator

    assert TorchApproximator.__bases__ == (TorchEstimator,)
    with pytest.warns(DeprecationWarning, match="TorchApproximator.*TorchEstimator"):
        instance = TorchApproximator()
    assert isinstance(instance, TorchEstimator)
