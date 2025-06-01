import pytest


@pytest.fixture()
def cosine_noise_schedule():
    from bayesflow.experimental.diffusion_model.schedules import CosineNoiseSchedule

    return CosineNoiseSchedule(min_log_snr=-12, max_log_snr=12, shift=0.1, weighting="likelihood_weighting")


@pytest.fixture()
def edm_noise_schedule():
    from bayesflow.experimental.diffusion_model.schedules import EDMNoiseSchedule

    return EDMNoiseSchedule(sigma_data=10.0, sigma_min=1e-5, sigma_max=85.0)


@pytest.fixture(
    params=["cosine_noise_schedule", "edm_noise_schedule"],
    scope="function",
)
def noise_schedule(request):
    return request.getfixturevalue(request.param)
