import numpy as np
import pytest


@pytest.fixture()
def batch_size():
    return 16


@pytest.fixture(params=[False, True], autouse=True)
def use_batched(request):
    return request.param


@pytest.fixture(params=[False, True], autouse=True)
def use_numpy(request):
    return request.param


@pytest.fixture(params=[False, True], autouse=True)
def use_squeezed(request):
    return request.param


@pytest.fixture()
def bernoulli_glm():
    from bayesflow.simulators import BernoulliGLM

    return BernoulliGLM()


@pytest.fixture()
def bernoulli_glm_raw():
    from bayesflow.simulators import BernoulliGLMRaw

    return BernoulliGLMRaw()


@pytest.fixture()
def gaussian_linear():
    from bayesflow.simulators import GaussianLinear

    return GaussianLinear()


@pytest.fixture()
def gaussian_linear_n_obs():
    from bayesflow.simulators import GaussianLinear

    return GaussianLinear(n_obs=5)


@pytest.fixture()
def gaussian_linear_uniform():
    from bayesflow.simulators import GaussianLinearUniform

    return GaussianLinearUniform()


@pytest.fixture()
def gaussian_linear_uniform_n_obs():
    from bayesflow.simulators import GaussianLinearUniform

    return GaussianLinearUniform(n_obs=5)


@pytest.fixture(
    params=["gaussian_linear", "gaussian_linear_n_obs", "gaussian_linear_uniform", "gaussian_linear_uniform_n_obs"]
)
def gaussian_linear_simulator(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def gaussian_mixture():
    from bayesflow.simulators import GaussianMixture

    return GaussianMixture()


@pytest.fixture()
def inverse_kinematics():
    from bayesflow.simulators import InverseKinematics

    return InverseKinematics()


@pytest.fixture()
def lotka_volterra():
    from bayesflow.simulators import LotkaVolterra

    return LotkaVolterra()


@pytest.fixture()
def sir():
    from bayesflow.simulators import SIR

    return SIR()


@pytest.fixture()
def slcp():
    from bayesflow.simulators import SLCP

    return SLCP()


@pytest.fixture()
def slcp_distractors():
    from bayesflow.simulators import SLCPDistractors

    return SLCPDistractors()


@pytest.fixture()
def composite_two_moons():
    from bayesflow.simulators import make_simulator

    def parameters():
        parameters = np.random.uniform(-1.0, 1.0, size=2)
        return dict(parameters=parameters)

    def observables(parameters):
        r = np.random.normal(0.1, 0.01)
        alpha = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi)
        x1 = -np.abs(parameters[0] + parameters[1]) / np.sqrt(2.0) + r * np.cos(alpha) + 0.25
        x2 = (-parameters[0] + parameters[1]) / np.sqrt(2.0) + r * np.sin(alpha)
        return dict(observables=np.stack([x1, x2]))

    return make_simulator([parameters, observables])


@pytest.fixture()
def two_moons():
    from bayesflow.simulators import TwoMoons

    return TwoMoons()


@pytest.fixture(
    params=[
        "composite_two_moons",
        "two_moons",
    ]
)
def two_moons_simulator(request):
    return request.getfixturevalue(request.param)


@pytest.fixture()
def composite_gaussian():
    from bayesflow.simulators import make_simulator

    def context():
        n = np.random.randint(10, 100)
        return dict(n=n)

    def prior():
        mu = np.random.normal(0, 1)
        return dict(mu=mu)

    def likelihood(mu, n):
        y = np.random.normal(mu, 1, n)
        return dict(y=y)

    return make_simulator([prior, likelihood], meta_fn=context)


@pytest.fixture()
def fixed_n():
    return 5


@pytest.fixture()
def fixed_mu():
    return 100


@pytest.fixture(
    params=[
        "bernoulli_glm",
        "bernoulli_glm_raw",
        "gaussian_linear",
        "gaussian_linear_n_obs",
        "gaussian_linear_uniform",
        "gaussian_linear_uniform_n_obs",
        "gaussian_mixture",
        "inverse_kinematics",
        "lotka_volterra",
        "sir",
        "slcp",
        "slcp_distractors",
        "composite_two_moons",
        "two_moons",
    ]
)
def simulator(request):
    return request.getfixturevalue(request.param)
