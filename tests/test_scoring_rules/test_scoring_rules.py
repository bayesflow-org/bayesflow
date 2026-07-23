import keras
import pytest


def test_logistic_score_get_config():
    from bayesflow.scoring_rules import LogisticScore

    rule = LogisticScore()
    config = rule.get_config()
    assert config["alpha"] == 0.0

    rule = LogisticScore(alpha=1.5)
    config = rule.get_config()
    assert config["alpha"] == 1.5


def test_exponential_score_leaky_get_config():
    from bayesflow.links import Leaky
    from bayesflow.scoring_rules import ExponentialScore

    rule = ExponentialScore(links={"logits": Leaky(power=2.0)})

    link = rule.get_link("logits")
    link_config = link.get_config()
    assert link_config["power"] == 2.0


def test_require_argument_k():
    from bayesflow.scoring_rules import NormedDifferenceScore

    with pytest.raises(TypeError) as excinfo:
        NormedDifferenceScore()

    assert "missing 1 required positional argument: 'k'" in str(excinfo)


def test_score_output(scoring_rule, random_conditions):
    if random_conditions is None:
        random_conditions = keras.ops.convert_to_tensor([[1.0, 1.0]])

    # Using random random_conditions also as targets for the purpose of this test.
    head_shapes = scoring_rule.get_head_shapes_from_target_shape(random_conditions.shape)
    estimates = {}
    for key, output_shape in head_shapes.items():
        link = scoring_rule.get_link(key)
        if hasattr(link, "compute_input_shape"):
            link_input_shape = link.compute_input_shape(output_shape)
        else:
            link_input_shape = output_shape
        dummy_input = keras.random.normal((random_conditions.shape[0],) + link_input_shape)
        estimates[key] = link(dummy_input)

    score = scoring_rule.score(estimates, random_conditions)

    assert score.ndim == 0


def test_mean_score_optimality(mean_score, random_conditions):
    if random_conditions is None:
        random_conditions = keras.ops.convert_to_tensor([[1.0]])

    key = "value"
    suboptimal_estimates = {key: keras.random.uniform(random_conditions.shape)}
    optimal_estimates = {key: random_conditions}

    suboptimal_score = mean_score.score(suboptimal_estimates, random_conditions)
    optimal_score = mean_score.score(optimal_estimates, random_conditions)

    assert suboptimal_score > optimal_score
    assert keras.ops.isclose(optimal_score, 0)


def test_unconditional_mvn(multivariate_normal_score):
    mean = keras.ops.convert_to_tensor([[0.0, 1.0]])
    covariance = keras.ops.convert_to_tensor([[[1.0, 0.0], [0.0, 1.0]]])
    multivariate_normal_score.sample((10,), mean, covariance)


def test_mixture_score_constructor_validation():
    from bayesflow.scoring_rules import MvNormalScore, MixtureScore

    with pytest.raises(ValueError, match="at least two"):
        MixtureScore(mvn1=MvNormalScore())


def test_mixture_score_sample_shape(mixture_of_multivariate_normal_scores):
    batch_size, dim = 4, 3
    mix = mixture_of_multivariate_normal_scores
    eye = keras.ops.broadcast_to(keras.ops.eye(dim)[None], (batch_size, dim, dim))
    estimates = {
        "mixture_logits": keras.ops.zeros((batch_size, 2)),
        "mvn1__mean": keras.ops.zeros((batch_size, dim)),
        "mvn1__precision_cholesky_factor": eye,
        "mvn2__mean": keras.ops.zeros((batch_size, dim)),
        "mvn2__precision_cholesky_factor": eye,
    }

    samples = mix.sample((batch_size,), **estimates)

    assert samples.shape == (batch_size, dim)


def test_mixture_score_set_temperature(mixture_of_multivariate_normal_scores):
    mixture_of_multivariate_normal_scores.set_temperature(2.5)
    assert float(mixture_of_multivariate_normal_scores.temperature) == pytest.approx(2.5)


def test_mixture_score_transformation_type_propagates_from_components(mixture_of_multivariate_normal_scores):
    mix = mixture_of_multivariate_normal_scores
    assert mix.TRANSFORMATION_TYPE["mixture_logits"] == "identity"
    assert mix.TRANSFORMATION_TYPE["mvn1__precision_cholesky_factor"] == "right_side_scale_inverse"
    assert mix.TRANSFORMATION_TYPE["mvn2__precision_cholesky_factor"] == "right_side_scale_inverse"


def test_mixture_score_serialization():
    from bayesflow.scoring_rules import MvNormalScore, MixtureScore
    from bayesflow.utils.serialization import serialize, deserialize

    original = MixtureScore(mvn1=MvNormalScore(), mvn2=MvNormalScore())
    restored = deserialize(serialize(original))

    assert isinstance(restored, MixtureScore)
    assert list(restored.components.keys()) == list(original.components.keys())


# --- ScoringRule base class ---


def test_scoring_rule_score_raises():
    from bayesflow.scoring_rules import ScoringRule

    rule = ScoringRule()
    with pytest.raises(NotImplementedError):
        rule.score({}, None, None)


def test_scoring_rule_get_head_shapes_raises():
    from bayesflow.scoring_rules import ScoringRule

    rule = ScoringRule()
    with pytest.raises(NotImplementedError):
        rule.get_head_shapes_from_target_shape((1, 2))


def test_scoring_rule_get_subnet_default():
    import keras
    from bayesflow.scoring_rules import ScoringRule

    rule = ScoringRule()
    subnet = rule.get_subnet("any_key")
    assert isinstance(subnet, keras.layers.Identity)


def test_scoring_rule_get_link_default():
    import keras
    from bayesflow.scoring_rules import ScoringRule

    rule = ScoringRule()
    link = rule.get_link("any_key")
    assert isinstance(link, keras.layers.Activation)


def test_scoring_rule_get_link_string():
    import keras
    from bayesflow.scoring_rules import ScoringRule

    rule = ScoringRule(links={"value": "relu"})
    link = rule.get_link("value")
    assert isinstance(link, keras.layers.Activation)


def test_scoring_rule_get_link_layer():
    import keras
    from bayesflow.scoring_rules import ScoringRule

    layer = keras.layers.Activation("sigmoid")
    rule = ScoringRule(links={"value": layer})
    assert rule.get_link("value") is layer


def test_scoring_rule_get_config_round_trip():
    from bayesflow.scoring_rules import BrierScore
    from bayesflow.utils.serialization import serialize, deserialize

    original = BrierScore()
    restored = deserialize(serialize(original))
    assert isinstance(restored, BrierScore)


# --- PolynomialScore ---


def test_polynomial_score_alpha_validation():
    from bayesflow.scoring_rules import PolynomialScore

    with pytest.raises(ValueError, match="greater than 1"):
        PolynomialScore(alpha=1.0)
    with pytest.raises(ValueError, match="greater than 1"):
        PolynomialScore(alpha=0.5)


def test_polynomial_score_with_weights():
    import keras
    from bayesflow.scoring_rules import PolynomialScore

    rule = PolynomialScore(alpha=2.0)
    targets = keras.ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])
    probs = keras.ops.softmax(keras.ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]]), axis=-1)
    # weights=[2, 0] → weighted_mean uses ops.mean(score * weight), so result = score[0]
    weights = keras.ops.convert_to_tensor([2.0, 0.0])
    score_weighted = rule.score({"probs": probs}, targets, weights=weights)
    score_first = rule.score({"probs": probs[:1]}, targets[:1])
    assert keras.ops.allclose(score_weighted, score_first, atol=1e-5)


def test_polynomial_score_get_config():
    from bayesflow.scoring_rules import PolynomialScore

    rule = PolynomialScore(alpha=3.0)
    config = rule.get_config()
    assert config["alpha"] == 3.0


# --- BrierScore ---


def test_brier_score_with_weights():
    import keras
    from bayesflow.scoring_rules import BrierScore

    rule = BrierScore()
    targets = keras.ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])
    probs = keras.ops.softmax(keras.ops.convert_to_tensor([[2.0, 0.0], [0.0, 2.0]]), axis=-1)
    weights = keras.ops.convert_to_tensor([2.0, 0.0])
    score_weighted = rule.score({"probs": probs}, targets, weights=weights)
    score_first = rule.score({"probs": probs[:1]}, targets[:1])
    assert keras.ops.allclose(score_weighted, score_first, atol=1e-5)


def test_brier_score_get_config_round_trip():
    from bayesflow.scoring_rules import BrierScore
    from bayesflow.utils.serialization import serialize, deserialize

    original = BrierScore()
    restored = deserialize(serialize(original))
    assert isinstance(restored, BrierScore)


def test_brier_score_optimal_at_true_probs():
    import keras
    from bayesflow.scoring_rules import BrierScore

    rule = BrierScore()
    targets = keras.ops.convert_to_tensor([[1.0, 0.0], [1.0, 0.0]])
    # perfect probs vs uniform
    perfect_probs = keras.ops.softmax(keras.ops.convert_to_tensor([[10.0, -10.0], [10.0, -10.0]]), axis=-1)
    uniform_probs = keras.ops.softmax(keras.ops.convert_to_tensor([[0.0, 0.0], [0.0, 0.0]]), axis=-1)
    assert rule.score({"probs": perfect_probs}, targets) < rule.score({"probs": uniform_probs}, targets)


def test_brier_score_is_polynomial_special_case():
    """BrierScore is the alpha=2 PolynomialScore, with alpha fixed and not serialized."""
    from bayesflow.scoring_rules import BrierScore, PolynomialScore

    rule = BrierScore()
    assert isinstance(rule, PolynomialScore)
    assert rule.alpha == 2.0
    # alpha is not a BrierScore parameter, so it must not leak into the config
    assert "alpha" not in rule.get_config()


def test_brier_score_reports_true_brier_value():
    """BrierScore reuses PolynomialScore's computation but reports the true Brier value."""
    import keras
    import numpy as np
    from bayesflow.scoring_rules import BrierScore, PolynomialScore

    targets = keras.ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])
    probs = keras.ops.softmax(keras.ops.convert_to_tensor([[2.0, 0.0], [0.0, 3.0]]), axis=-1)
    brier = BrierScore().score({"probs": probs}, targets)

    # Exact (mean) Brier score sum((p - y)^2).
    probs_np = np.asarray(probs)
    expected = np.mean(np.sum((probs_np - np.asarray(targets)) ** 2, axis=-1))
    assert keras.ops.allclose(brier, expected, atol=1e-6)

    # ...recovered from the alpha=2 Tsallis score via the affine map S_Brier = 2 S_poly + 1.
    poly = PolynomialScore(alpha=2.0).score({"probs": probs}, targets)
    assert keras.ops.allclose(brier, 2.0 * poly + 1.0, atol=1e-6)


# --- LogisticScore ---


def test_logistic_score_with_weights():
    import keras
    from bayesflow.scoring_rules import LogisticScore

    rule = LogisticScore()
    targets = keras.ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])
    estimates = keras.ops.convert_to_tensor([[1.0, -1.0], [-1.0, 1.0]])
    weights = keras.ops.convert_to_tensor([2.0, 0.0])
    score_weighted = rule.score({"logits": estimates}, targets, weights=weights)
    score_first = rule.score({"logits": estimates[:1]}, targets[:1])
    assert keras.ops.allclose(score_weighted, score_first, atol=1e-5)


def test_logistic_score_get_config_round_trip():
    from bayesflow.scoring_rules import LogisticScore
    from bayesflow.utils.serialization import serialize, deserialize

    original = LogisticScore()
    restored = deserialize(serialize(original))
    assert isinstance(restored, LogisticScore)


# --- ExponentialScore ---


def test_exponential_score_with_weights():
    import keras
    from bayesflow.scoring_rules import ExponentialScore

    rule = ExponentialScore()
    targets = keras.ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])
    estimates = keras.ops.convert_to_tensor([[1.0, -1.0], [-1.0, 1.0]])
    weights = keras.ops.convert_to_tensor([2.0, 0.0])
    score_weighted = rule.score({"logits": estimates}, targets, weights=weights)
    score_first = rule.score({"logits": estimates[:1]}, targets[:1])
    assert keras.ops.allclose(score_weighted, score_first, atol=1e-5)


def test_exponential_score_clipping_no_overflow():
    import keras
    from bayesflow.scoring_rules import ExponentialScore

    rule = ExponentialScore()
    targets = keras.ops.convert_to_tensor([[1.0, 0.0]])
    large_estimates = keras.ops.convert_to_tensor([[0.0, 1000.0]])
    score = rule.score({"logits": large_estimates}, targets)
    assert keras.ops.isfinite(score)


def test_exponential_score_get_config_round_trip():
    from bayesflow.scoring_rules import ExponentialScore
    from bayesflow.utils.serialization import serialize, deserialize

    original = ExponentialScore()
    restored = deserialize(serialize(original))
    assert isinstance(restored, ExponentialScore)


# --- LogisticScore (power form) ---


def test_logistic_score_alpha_validation():
    from bayesflow.scoring_rules import LogisticScore

    with pytest.raises(ValueError, match="non-negative"):
        LogisticScore(alpha=-1.0)


def test_power_logistic_score_clipping_no_overflow():
    import keras
    from bayesflow.scoring_rules import LogisticScore

    rule = LogisticScore(alpha=2.0)
    targets = keras.ops.convert_to_tensor([[1.0, 0.0]])
    large_estimates = keras.ops.convert_to_tensor([[0.0, 1000.0]])
    score = rule.score({"logits": large_estimates}, targets)
    assert keras.ops.isfinite(score)


def test_power_logistic_score_get_config_round_trip():
    from bayesflow.scoring_rules import LogisticScore
    from bayesflow.utils.serialization import serialize, deserialize

    original = LogisticScore(alpha=2.0)
    restored = deserialize(serialize(original))
    assert isinstance(restored, LogisticScore)
    assert restored.alpha == 2.0
