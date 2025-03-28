import keras
import pytest


def test_require_argument_k():
    from bayesflow.scores import NormedDifferenceScore

    with pytest.raises(TypeError) as excinfo:
        NormedDifferenceScore()

    assert "missing 1 required positional argument: 'k'" in str(excinfo)


def test_score_output(scoring_rule, random_conditions):
    if random_conditions is None:
        random_conditions = keras.ops.convert_to_tensor([[1.0, 1.0]])

    # Using random random_conditions also as targets for the purpose of this test.
    head_shapes = scoring_rule.get_head_shapes_from_target_shape(random_conditions.shape)
    print(scoring_rule.get_config())
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


def test_unconditional_mvn_value_error(multivariate_normal_score):
    mean = keras.ops.convert_to_tensor([0.0, 1.0])
    covariance = keras.ops.convert_to_tensor([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError):
        multivariate_normal_score.sample((10,), mean, covariance)
