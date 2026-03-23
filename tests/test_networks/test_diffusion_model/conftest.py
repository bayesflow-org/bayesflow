import keras
import pytest


# ---------------------------------------------------------------------------
# Noise schedules
# ---------------------------------------------------------------------------


@pytest.fixture()
def cosine_noise_schedule():
    from bayesflow.networks.inference.diffusion.schedules import CosineNoiseSchedule

    return CosineNoiseSchedule(min_log_snr=-12, max_log_snr=12, shift=0.1)


@pytest.fixture()
def edm_noise_schedule():
    from bayesflow.networks.inference.diffusion.schedules import EDMNoiseSchedule

    return EDMNoiseSchedule(sigma_data=10.0, sigma_min=1e-5, sigma_max=85.0)


@pytest.fixture(params=["cosine_noise_schedule", "edm_noise_schedule"], scope="function")
def noise_schedule(request):
    return request.getfixturevalue(request.param)


# ---------------------------------------------------------------------------
# Shapes
# ---------------------------------------------------------------------------


@pytest.fixture(params=[2], scope="session")
def batch_size(request):
    return request.param


@pytest.fixture(params=[2, 5], scope="session")
def xz_dim(request):
    return request.param


@pytest.fixture(params=[None, 3], scope="session")
def cond_dim(request):
    return request.param


@pytest.fixture(scope="session")
def random_samples(batch_size, xz_dim):
    return keras.random.normal((batch_size, xz_dim))


@pytest.fixture(scope="session")
def random_conditions(batch_size, cond_dim):
    if cond_dim is None:
        return None
    return keras.random.normal((batch_size, cond_dim))


# ---------------------------------------------------------------------------
# DiffusionModel variants
# ---------------------------------------------------------------------------


@pytest.fixture(
    params=[
        dict(noise_schedule="edm", prediction_type="F"),
        dict(noise_schedule="cosine", prediction_type="velocity"),
        dict(noise_schedule="edm", prediction_type="potential"),
    ],
    ids=lambda d: f"{d['noise_schedule']}_{d['prediction_type']}",
)
def diffusion_model(request):
    from bayesflow.networks import DiffusionModel

    return DiffusionModel(subnet_kwargs=dict(widths=(8, 8)), **request.param)


@pytest.fixture()
def diffusion_model_with_masking():
    from bayesflow.networks import DiffusionModel

    return DiffusionModel(
        subnet_kwargs=dict(widths=(8, 8)),
        drop_cond_prob=0.1,
        drop_target_prob=0.5,
    )


@pytest.fixture
def simple_diffusion_model():
    """Create a simple diffusion model for testing compositional sampling."""
    from bayesflow.networks import DiffusionModel

    return DiffusionModel(
        subnet_kwargs={"widths": (32, 32)},
    )


@pytest.fixture
def simple_compositional_diffusion_model():
    """Create a simple diffusion model for testing compositional sampling."""
    from bayesflow.networks import CompositionalDiffusionModel

    return CompositionalDiffusionModel(
        subnet_kwargs={"widths": (32, 32)},
    )


@pytest.fixture
def compositional_conditions():
    """Create test conditions for compositional sampling."""
    batch_size = 2
    n_compositional = 3
    condition_dim = 5

    return keras.random.normal((batch_size, n_compositional, condition_dim))


@pytest.fixture
def compositional_state():
    """Create test state for compositional sampling."""
    batch_size = 2
    param_dim = 3

    return keras.random.normal((batch_size, param_dim))


@pytest.fixture
def mock_prior_score():
    """Create a mock prior score function for testing."""

    def prior_score_fn(theta):
        # Simple quadratic prior: -0.5 * ||theta||^2
        return -theta

    return prior_score_fn
