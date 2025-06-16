import pytest

from bayesflow.networks import MLP


@pytest.fixture()
def diffusion_model_edm_F():
    from bayesflow.experimental import DiffusionModel

    return DiffusionModel(
        subnet=MLP([8, 8]),
        integrate_kwargs={"method": "rk45", "steps": 10},
        noise_schedule="edm",
        prediction_type="F",
    )


@pytest.fixture()
def diffusion_model_edm_velocity():
    from bayesflow.experimental import DiffusionModel

    return DiffusionModel(
        subnet=MLP([8, 8]),
        integrate_kwargs={"method": "rk45", "steps": 10},
        noise_schedule="edm",
        prediction_type="velocity",
    )


@pytest.fixture()
def diffusion_model_edm_noise():
    from bayesflow.experimental import DiffusionModel

    return DiffusionModel(
        subnet=MLP([8, 8]),
        integrate_kwargs={"method": "rk45", "steps": 10},
        noise_schedule="edm",
        prediction_type="noise",
    )


@pytest.fixture()
def diffusion_model_cosine_F():
    from bayesflow.experimental import DiffusionModel

    return DiffusionModel(
        subnet=MLP([8, 8]),
        integrate_kwargs={"method": "rk45", "steps": 10},
        noise_schedule="cosine",
        prediction_type="F",
    )


@pytest.fixture()
def diffusion_model_cosine_velocity():
    from bayesflow.experimental import DiffusionModel

    return DiffusionModel(
        subnet=MLP([8, 8]),
        integrate_kwargs={"method": "rk45", "steps": 10},
        noise_schedule="cosine",
        prediction_type="velocity",
    )


@pytest.fixture()
def diffusion_model_cosine_noise():
    from bayesflow.experimental import DiffusionModel

    return DiffusionModel(
        subnet=MLP([8, 8]),
        integrate_kwargs={"method": "rk45", "steps": 10},
        noise_schedule="cosine",
        prediction_type="noise",
    )


@pytest.fixture()
def flow_matching():
    from bayesflow.networks import FlowMatching

    return FlowMatching(
        subnet=MLP([8, 8]),
        integrate_kwargs={"method": "rk45", "steps": 10},
    )


@pytest.fixture()
def consistency_model():
    from bayesflow.networks import ConsistencyModel

    return ConsistencyModel(total_steps=100, subnet=MLP([8, 8]))


@pytest.fixture()
def affine_coupling_flow():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(
        depth=2, subnet="mlp", subnet_kwargs=dict(widths=[8, 8]), transform="affine", transform_kwargs=dict(clamp=1.8)
    )


@pytest.fixture()
def spline_coupling_flow():
    from bayesflow.networks import CouplingFlow

    return CouplingFlow(
        depth=2, subnet="mlp", subnet_kwargs=dict(widths=[8, 8]), transform="spline", transform_kwargs=dict(bins=8)
    )


@pytest.fixture()
def free_form_flow():
    from bayesflow.experimental import FreeFormFlow

    return FreeFormFlow(encoder_subnet=MLP([16, 16]), decoder_subnet=MLP([16, 16]))


@pytest.fixture()
def typical_point_inference_network():
    from bayesflow.networks import PointInferenceNetwork
    from bayesflow.scores import MeanScore, MedianScore, QuantileScore, MultivariateNormalScore

    return PointInferenceNetwork(
        scores=dict(
            mean=MeanScore(),
            median=MedianScore(),
            quantiles=QuantileScore([0.1, 0.2, 0.5, 0.65]),
            mvn=MultivariateNormalScore(),  # currently not stable
        )
    )


@pytest.fixture()
def typical_point_inference_network_subnet():
    from bayesflow.networks import PointInferenceNetwork
    from bayesflow.scores import MeanScore, MedianScore, QuantileScore, MultivariateNormalScore

    subnet = MLP([16, 8])

    return PointInferenceNetwork(
        scores=dict(
            mean=MeanScore(subnets=dict(value=subnet)),
            median=MedianScore(subnets=dict(value=subnet)),
            quantiles=QuantileScore(subnets=dict(value=subnet)),
            mvn=MultivariateNormalScore(subnets=dict(mean=subnet, covariance=subnet)),
        ),
        subnet=subnet,
    )


@pytest.fixture(
    params=[
        "typical_point_inference_network",
        "affine_coupling_flow",
        "spline_coupling_flow",
        "flow_matching",
        "free_form_flow",
        "consistency_model",
        pytest.param("diffusion_model_edm_F"),
        pytest.param("diffusion_model_edm_noise", marks=[pytest.mark.slow]),
        pytest.param("diffusion_model_cosine_velocity", marks=[pytest.mark.slow]),
        pytest.param("diffusion_model_cosine_F", marks=[pytest.mark.slow]),
        pytest.param("diffusion_model_cosine_noise", marks=[pytest.mark.slow]),
        pytest.param("diffusion_model_cosine_velocity", marks=[pytest.mark.slow]),
    ],
    scope="function",
)
def inference_network(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "typical_point_inference_network_subnet",
        "coupling_flow_subnet",
        "flow_matching_subnet",
        "free_form_flow_subnet",
    ],
    scope="function",
)
def inference_network_subnet(request):
    return request.getfixturevalue(request.param)


@pytest.fixture(
    params=[
        "affine_coupling_flow",
        "spline_coupling_flow",
        "flow_matching",
        "free_form_flow",
        "consistency_model",
        pytest.param("diffusion_model_edm_F"),
        pytest.param(
            "diffusion_model_edm_noise",
            marks=[
                pytest.mark.slow,
                pytest.mark.skip("noise predicition not testable without prior training for numerical reasons."),
            ],
        ),
        pytest.param("diffusion_model_cosine_velocity", marks=[pytest.mark.slow]),
        pytest.param(
            "diffusion_model_cosine_F",
            marks=[
                pytest.mark.slow,
                pytest.mark.skip("skip to reduce load on CI."),
            ],
        ),
        pytest.param(
            "diffusion_model_cosine_noise",
            marks=[
                pytest.mark.slow,
                pytest.mark.skip("noise predicition not testable without prior training for numerical reasons."),
            ],
        ),
        pytest.param(
            "diffusion_model_cosine_velocity",
            marks=[
                pytest.mark.slow,
                pytest.mark.skip("skip to reduce load on CI."),
            ],
        ),
    ],
    scope="function",
)
def generative_inference_network(request):
    return request.getfixturevalue(request.param)
