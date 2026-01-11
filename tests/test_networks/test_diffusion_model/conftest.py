import pytest
import keras


@pytest.fixture()
def cosine_noise_schedule():
    from bayesflow.networks.diffusion_model.schedules import CosineNoiseSchedule

    return CosineNoiseSchedule(min_log_snr=-12, max_log_snr=12, shift=0.1, weighting="sigmoid")


@pytest.fixture()
def edm_noise_schedule():
    from bayesflow.networks.diffusion_model.schedules import EDMNoiseSchedule

    return EDMNoiseSchedule(sigma_data=10.0, sigma_min=1e-5, sigma_max=85.0)


@pytest.fixture(
    params=["cosine_noise_schedule", "edm_noise_schedule"],
    scope="function",
)
def noise_schedule(request):
    return request.getfixturevalue(request.param)


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
def simple_compositional_diffusion_model():
    """Create a simple diffusion model for testing compositional sampling."""
    from bayesflow.networks.diffusion_model import CompositionalDiffusionModel
    from bayesflow.networks import MLP

    return CompositionalDiffusionModel(
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
