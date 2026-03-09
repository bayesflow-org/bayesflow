import keras
import pytest


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
    from bayesflow.scoring_rules import MvNormalScore, MixtureScore, MeanScore

    with pytest.raises(ValueError, match="at least two"):
        MixtureScore(mvn1=MvNormalScore())

    with pytest.raises(TypeError, match="ParametricDistributionScore"):
        MixtureScore(components={"a": MvNormalScore(), "b": MeanScore()})


def test_mixture_score_sample_shape(mixture_of_multivariate_normal_scores):
    batch_size, num_samples, dim = 4, 10, 3
    mix = mixture_of_multivariate_normal_scores
    eye = keras.ops.broadcast_to(keras.ops.eye(dim)[None], (batch_size, dim, dim))
    estimates = {
        "mixture_logits": keras.ops.zeros((batch_size, 2)),
        "mvn1__mean": keras.ops.zeros((batch_size, dim)),
        "mvn1__precision_cholesky_factor": eye,
        "mvn2__mean": keras.ops.zeros((batch_size, dim)),
        "mvn2__precision_cholesky_factor": eye,
    }

    samples = mix.sample((batch_size, num_samples), **estimates)

    assert samples.shape == (batch_size, num_samples, dim)


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
