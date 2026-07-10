import keras
import pytest


def test_is_pmp_rule_pmp_rules():
    from bayesflow.scoring_rules import CrossEntropyScore, BrierScore, PolynomialScore

    assert CrossEntropyScore().is_pmp_rule is True
    assert BrierScore().is_pmp_rule is True
    assert PolynomialScore(alpha=2.0).is_pmp_rule is True


def test_is_pmp_rule_bf_rules():
    from bayesflow.links import Leaky
    from bayesflow.scoring_rules import ExponentialScore, LogisticScore

    assert ExponentialScore().is_pmp_rule is False
    assert ExponentialScore(scale=2.0).is_pmp_rule is False
    assert ExponentialScore(links={"log_bayes_factors": Leaky(power=2.0)}).is_pmp_rule is False
    assert LogisticScore().is_pmp_rule is False
    assert LogisticScore(alpha=1.0).is_pmp_rule is False


def test_to_bayes_factors_exponential_is_identity():
    from bayesflow.scoring_rules import ExponentialScore

    rule = ExponentialScore()
    f = keras.ops.convert_to_tensor([[1.0, -2.0, 0.5]])
    result = rule.to_bayes_factors(f)
    assert keras.ops.allclose(result, f)


def test_to_bayes_factors_scaled_exponential():
    from bayesflow.scoring_rules import ExponentialScore

    scale = 3.0
    rule = ExponentialScore(scale=scale)
    f = keras.ops.convert_to_tensor([[1.0, -2.0]])
    result = rule.to_bayes_factors(f)
    assert keras.ops.allclose(result, f * scale)


def test_to_bayes_factors_plain_logistic_is_identity():
    """LogisticScore.to_bayes_factors is the identity for alpha = 0 (plain rule)."""
    from bayesflow.scoring_rules import LogisticScore

    rule = LogisticScore()
    f = keras.ops.convert_to_tensor([[0.5, -0.5]])
    result = rule.to_bayes_factors(f)
    assert keras.ops.allclose(result, f)


def test_to_bayes_factors_power_logistic():
    from bayesflow.scoring_rules import LogisticScore

    alpha = 2.0
    rule = LogisticScore(alpha=alpha)
    f = keras.ops.convert_to_tensor([[1.0, -1.0]])
    result = rule.to_bayes_factors(f)
    assert keras.ops.allclose(result, f * (alpha + 1))


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

    rule = ExponentialScore(links={"log_bayes_factors": Leaky(power=2.0)})
    config = rule.get_config()
    assert config["scale"] == 1.0

    link = rule.get_link("log_bayes_factors")
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
    logits = keras.ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])
    # weights=[2, 0] → weighted_mean uses ops.mean(score * weight), so result = score[0]
    weights = keras.ops.convert_to_tensor([2.0, 0.0])
    score_weighted = rule.score({"logits": logits}, targets, weights=weights)
    score_first = rule.score({"logits": logits[:1]}, targets[:1])
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
    logits = keras.ops.convert_to_tensor([[2.0, 0.0], [0.0, 2.0]])
    weights = keras.ops.convert_to_tensor([2.0, 0.0])
    score_weighted = rule.score({"logits": logits}, targets, weights=weights)
    score_first = rule.score({"logits": logits[:1]}, targets[:1])
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
    # perfect logits vs random
    perfect_logits = keras.ops.convert_to_tensor([[10.0, -10.0], [10.0, -10.0]])
    random_logits = keras.ops.convert_to_tensor([[0.0, 0.0], [0.0, 0.0]])
    assert rule.score({"logits": perfect_logits}, targets) < rule.score({"logits": random_logits}, targets)


def test_brier_score_is_polynomial_special_case():
    """BrierScore is the alpha=2 PolynomialScore, with alpha fixed and not serialized."""
    from bayesflow.scoring_rules import BrierScore, PolynomialScore

    rule = BrierScore()
    assert isinstance(rule, PolynomialScore)
    assert rule.alpha == 2.0
    # alpha is not a BrierScore parameter, so it must not leak into the config
    assert "alpha" not in rule.get_config()


def test_brier_score_reports_true_brier_value():
    """BrierScore.score returns the exact Brier value sum((p - y)^2), not the Tsallis score."""
    import keras
    import numpy as np
    from bayesflow.scoring_rules import BrierScore

    rule = BrierScore()
    targets = keras.ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])
    logits = keras.ops.convert_to_tensor([[2.0, 0.0], [0.0, 3.0]])
    probs = np.asarray(keras.ops.softmax(logits, axis=-1))
    expected = np.mean(np.sum((probs - np.asarray(targets)) ** 2, axis=-1))
    assert keras.ops.allclose(rule.score({"logits": logits}, targets), expected, atol=1e-6)


# --- LogisticScore ---


def test_logistic_score_with_weights():
    import keras
    from bayesflow.scoring_rules import LogisticScore

    rule = LogisticScore()
    targets = keras.ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])
    estimates = keras.ops.convert_to_tensor([[1.0, -1.0], [-1.0, 1.0]])
    weights = keras.ops.convert_to_tensor([2.0, 0.0])
    score_weighted = rule.score({"log_bayes_factors": estimates[:, 1:]}, targets, weights=weights)
    score_first = rule.score({"log_bayes_factors": estimates[:1, 1:]}, targets[:1])
    assert keras.ops.allclose(score_weighted, score_first, atol=1e-5)


def test_logistic_score_get_config_round_trip():
    from bayesflow.scoring_rules import LogisticScore
    from bayesflow.utils.serialization import serialize, deserialize

    original = LogisticScore()
    restored = deserialize(serialize(original))
    assert isinstance(restored, LogisticScore)


# --- ExponentialScore ---


def test_exponential_score_scale_validation():
    from bayesflow.scoring_rules import ExponentialScore

    with pytest.raises(ValueError, match="positive"):
        ExponentialScore(scale=0.0)
    with pytest.raises(ValueError, match="positive"):
        ExponentialScore(scale=-1.0)


def test_exponential_score_with_weights():
    import keras
    from bayesflow.scoring_rules import ExponentialScore

    rule = ExponentialScore()
    targets = keras.ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])
    estimates = keras.ops.convert_to_tensor([[1.0], [-1.0]])
    weights = keras.ops.convert_to_tensor([2.0, 0.0])
    score_weighted = rule.score({"log_bayes_factors": estimates}, targets, weights=weights)
    score_first = rule.score({"log_bayes_factors": estimates[:1]}, targets[:1])
    assert keras.ops.allclose(score_weighted, score_first, atol=1e-5)


def test_exponential_score_clipping_no_overflow():
    import keras
    from bayesflow.scoring_rules import ExponentialScore

    rule = ExponentialScore()
    targets = keras.ops.convert_to_tensor([[1.0, 0.0]])
    large_estimates = keras.ops.convert_to_tensor([[1000.0]])
    score = rule.score({"log_bayes_factors": large_estimates}, targets)
    assert keras.ops.isfinite(score)


def test_exponential_score_get_config_round_trip():
    from bayesflow.scoring_rules import ExponentialScore
    from bayesflow.utils.serialization import serialize, deserialize

    original = ExponentialScore(scale=2.0)
    restored = deserialize(serialize(original))
    assert isinstance(restored, ExponentialScore)
    assert restored.scale == 2.0


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
    large_estimates = keras.ops.convert_to_tensor([[1000.0]])
    score = rule.score({"log_bayes_factors": large_estimates}, targets)
    assert keras.ops.isfinite(score)


def test_power_logistic_score_get_config_round_trip():
    from bayesflow.scoring_rules import LogisticScore
    from bayesflow.utils.serialization import serialize, deserialize

    original = LogisticScore(alpha=2.0)
    restored = deserialize(serialize(original))
    assert isinstance(restored, LogisticScore)
    assert restored.alpha == 2.0
