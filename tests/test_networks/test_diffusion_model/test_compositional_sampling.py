import keras
import pytest


def test_compositional_score_shape(
    simple_compositional_diffusion_model, compositional_state, compositional_conditions, mock_prior_score
):
    """Test that compositional score returns correct shapes."""
    # Build the model
    state_shape = keras.ops.shape(compositional_state)
    conditions_shape = keras.ops.shape(compositional_conditions)
    simple_compositional_diffusion_model.build(state_shape, conditions_shape)
    simple_compositional_diffusion_model.compositional_bridge_d0 = 1
    simple_compositional_diffusion_model.compositional_bridge_d1 = 0.1

    time = 0.5

    score = simple_compositional_diffusion_model.compositional_score(
        xz=compositional_state,
        time=time,
        conditions=compositional_conditions,
        compute_prior_score=mock_prior_score,
        training=False,
    )

    expected_shape = keras.ops.shape(compositional_state)
    actual_shape = keras.ops.shape(score)

    assert keras.ops.all(keras.ops.equal(expected_shape, actual_shape)), (
        f"Expected shape {expected_shape}, got {actual_shape}"
    )


def test_compositional_score_no_conditions_raises_error(
    simple_compositional_diffusion_model, compositional_state, mock_prior_score
):
    """Test that compositional score raises error when conditions is None."""
    simple_compositional_diffusion_model.build(keras.ops.shape(compositional_state), None)

    with pytest.raises(ValueError, match="Conditions are required for compositional sampling"):
        simple_compositional_diffusion_model.compositional_score(
            xz=compositional_state, time=0.5, conditions=None, compute_prior_score=mock_prior_score, training=False
        )


def test_inverse_compositional_basic(
    simple_compositional_diffusion_model, compositional_state, compositional_conditions, mock_prior_score
):
    """Test basic compositional inverse sampling."""
    state_shape = keras.ops.shape(compositional_state)
    conditions_shape = keras.ops.shape(compositional_conditions)
    simple_compositional_diffusion_model.build(state_shape, conditions_shape)

    # Test inverse sampling
    result = simple_compositional_diffusion_model._inverse_compositional(
        z=compositional_state,
        conditions=compositional_conditions,
        compute_prior_score=mock_prior_score,
        density=False,
        training=False,
        method="euler_maruyama",
        steps=5,
        start_time=1.0,
        stop_time=0.0,
        mini_batch_size=1,
    )

    expected_shape = keras.ops.shape(compositional_state)
    actual_shape = keras.ops.shape(result)

    assert keras.ops.all(keras.ops.equal(expected_shape, actual_shape)), (
        f"Expected shape {expected_shape}, got {actual_shape}"
    )
