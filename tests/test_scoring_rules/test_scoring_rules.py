import keras
import pytest


def test_is_pmp_rule_pmp_rules():
    from bayesflow.scoring_rules import CrossEntropyScore, BrierScore, PolynomialScore

    assert CrossEntropyScore().is_pmp_rule is True
    assert BrierScore().is_pmp_rule is True
    assert PolynomialScore(alpha=2.0).is_pmp_rule is True


def test_is_pmp_rule_bf_rules():
    from bayesflow.scoring_rules import ExponentialScore, LogisticScore

    assert ExponentialScore().is_pmp_rule is False
    assert ExponentialScore(scale=2.0).is_pmp_rule is False
    assert ExponentialScore(leaky=2.0).is_pmp_rule is False
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


def test_to_bayes_factors_base_class_is_identity():
    """ScoringRule.to_bayes_factors is the identity by default."""
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

    rule_log = LogisticScore()
    config_log = rule_log.get_config()
    assert config_log["alpha"] is None

    rule_pow = LogisticScore(alpha=1.5)
    config_pow = rule_pow.get_config()
    assert config_pow["alpha"] == 1.5


def test_exponential_score_leaky_get_config():
    from bayesflow.scoring_rules import ExponentialScore

    rule = ExponentialScore(leaky=2.0)
    config = rule.get_config()
    assert config["leaky"] == 2.0
    assert config["scale"] == 1.0

    # _LeakyLink.get_config() is exercised via get_link
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
