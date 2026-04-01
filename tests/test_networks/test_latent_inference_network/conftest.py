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


# --- Encoder / Decoder fixtures ---


@pytest.fixture()
def encoder(latent_dim):
    from bayesflow.networks import Encoder

    return Encoder(latent_dim=latent_dim)


@pytest.fixture()
def encoder_auto():
    from bayesflow.networks import Encoder

    return Encoder(latent_dim="auto")


@pytest.fixture()
def decoder():
    from bayesflow.networks import Decoder

    return Decoder()


# --- LatentInferenceNetwork fixtures ---


@pytest.fixture()
def lin_with_diffusion(latent_dim):
    from bayesflow.networks import DiffusionModel, LatentInferenceNetwork

    return LatentInferenceNetwork(
        inference_network=DiffusionModel(subnet_kwargs=dict(widths=(8, 8))),
        latent_dim=latent_dim,
    )


@pytest.fixture()
def lin_with_flow_matching(latent_dim):
    from bayesflow.networks import FlowMatching, LatentInferenceNetwork

    return LatentInferenceNetwork(
        inference_network=FlowMatching(subnet_kwargs=dict(widths=(8, 8))),
        latent_dim=latent_dim,
    )
