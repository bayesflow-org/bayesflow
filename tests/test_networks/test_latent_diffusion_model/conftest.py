import keras
import pytest


@pytest.fixture(params=[2], scope="session")
def batch_size(request):
    return request.param


@pytest.fixture(params=[4, 8], scope="session")
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


@pytest.fixture(params=[2, 4], scope="session")
def latent_dim(request):
    return request.param


@pytest.fixture(
    params=[
        dict(noise_schedule="cosine"),
        dict(noise_schedule="edm"),
    ],
    ids=["cosine", "edm"],
)
def latent_diffusion_model(request, latent_dim):
    from bayesflow.networks import LatentDiffusionModel

    return LatentDiffusionModel(
        latent_dim=latent_dim,
        diffusion_subnet_kwargs=dict(widths=(8, 8)),
        **request.param,
    )


@pytest.fixture()
def latent_diffusion_model_with_flow_matching(latent_dim):
    from bayesflow.networks import FlowMatching, LatentDiffusionModel

    return LatentDiffusionModel(
        latent_dim=latent_dim,
        inference_network=FlowMatching(subnet_kwargs=dict(widths=(8, 8))),
    )
