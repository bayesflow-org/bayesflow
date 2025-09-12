import keras
import pytest


@pytest.fixture
def simple_diffusion_model():
    """Create a simple diffusion model for testing compositional sampling."""
    from bayesflow.networks.diffusion_model import DiffusionModel
    from bayesflow.networks import MLP

    return DiffusionModel(
        subnet=MLP(widths=[32, 32]),
        noise_schedule="cosine",
        prediction_type="noise",
        loss_type="noise",
    )


@pytest.fixture
def compositional_conditions():
    """Create test conditions for compositional sampling."""
    batch_size = 2
    n_compositional = 3
    n_samples = 4
    condition_dim = 5

    return keras.random.normal((batch_size, n_compositional, n_samples, condition_dim))


@pytest.fixture
def compositional_state():
    """Create test state for compositional sampling."""
    batch_size = 2
    n_samples = 4
    param_dim = 3

    return keras.random.normal((batch_size, n_samples, param_dim))


@pytest.fixture
def mock_prior_score():
    """Create a mock prior score function for testing."""

    def prior_score_fn(theta):
        # Simple quadratic prior: -0.5 * ||theta||^2
        return -theta

    return prior_score_fn


def test_compositional_score_shape(
    simple_diffusion_model, compositional_state, compositional_conditions, mock_prior_score
):
    """Test that compositional score returns correct shapes."""
    # Build the model
    state_shape = keras.ops.shape(compositional_state)
    conditions_shape = keras.ops.shape(compositional_conditions)
    simple_diffusion_model.build(state_shape, conditions_shape)

    time = 0.5

    score = simple_diffusion_model.compositional_score(
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


def test_compositional_score_no_conditions_raises_error(simple_diffusion_model, compositional_state, mock_prior_score):
    """Test that compositional score raises error when conditions is None."""
    simple_diffusion_model.build(keras.ops.shape(compositional_state), None)

    with pytest.raises(ValueError, match="Conditions are required for compositional sampling"):
        simple_diffusion_model.compositional_score(
            xz=compositional_state, time=0.5, conditions=None, compute_prior_score=mock_prior_score, training=False
        )


def test_inverse_compositional_basic(
    simple_diffusion_model, compositional_state, compositional_conditions, mock_prior_score
):
    """Test basic compositional inverse sampling."""
    state_shape = keras.ops.shape(compositional_state)
    conditions_shape = keras.ops.shape(compositional_conditions)
    simple_diffusion_model.build(state_shape, conditions_shape)

    # Test inverse sampling with ODE method
    result = simple_diffusion_model._inverse_compositional(
        z=compositional_state,
        conditions=compositional_conditions,
        compute_prior_score=mock_prior_score,
        density=False,
        training=False,
        method="euler",
        steps=5,
        start_time=1.0,
        stop_time=0.0,
    )

    expected_shape = keras.ops.shape(compositional_state)
    actual_shape = keras.ops.shape(result)

    assert keras.ops.all(keras.ops.equal(expected_shape, actual_shape)), (
        f"Expected shape {expected_shape}, got {actual_shape}"
    )


def test_inverse_compositional_euler_maruyama_with_corrector(
    simple_diffusion_model, compositional_state, compositional_conditions, mock_prior_score
):
    """Test compositional inverse sampling with Euler-Maruyama and corrector steps."""
    state_shape = keras.ops.shape(compositional_state)
    conditions_shape = keras.ops.shape(compositional_conditions)
    simple_diffusion_model.build(state_shape, conditions_shape)

    result = simple_diffusion_model._inverse_compositional(
        z=compositional_state,
        conditions=compositional_conditions,
        compute_prior_score=mock_prior_score,
        density=False,
        training=False,
        method="euler_maruyama",
        steps=5,
        corrector_steps=2,
        start_time=1.0,
        stop_time=0.0,
    )

    expected_shape = keras.ops.shape(compositional_state)
    actual_shape = keras.ops.shape(result)

    assert keras.ops.all(keras.ops.equal(expected_shape, actual_shape)), (
        f"Expected shape {expected_shape}, got {actual_shape}"
    )


@pytest.mark.parametrize("noise_schedule_name", ["cosine", "edm"])
def test_compositional_sampling_with_different_schedules(
    noise_schedule_name, compositional_state, compositional_conditions, mock_prior_score
):
    """Test compositional sampling with different noise schedules."""
    from bayesflow.networks.diffusion_model import DiffusionModel
    from bayesflow.networks import MLP

    diffusion_model = DiffusionModel(
        subnet=MLP(widths=[32, 32]),
        noise_schedule=noise_schedule_name,
        prediction_type="noise",
        loss_type="noise",
    )

    state_shape = keras.ops.shape(compositional_state)
    conditions_shape = keras.ops.shape(compositional_conditions)
    diffusion_model.build(state_shape, conditions_shape)

    score = diffusion_model.compositional_score(
        xz=compositional_state,
        time=0.5,
        conditions=compositional_conditions,
        compute_prior_score=mock_prior_score,
        training=False,
    )

    expected_shape = keras.ops.shape(compositional_state)
    actual_shape = keras.ops.shape(score)

    assert keras.ops.all(keras.ops.equal(expected_shape, actual_shape)), (
        f"Expected shape {expected_shape}, got {actual_shape}"
    )
