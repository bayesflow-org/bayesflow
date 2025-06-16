import pytest
from pathlib import Path


@pytest.fixture(autouse=True, scope="session")
def mode(request):
    mode = request.config.getoption("--mode")
    if not mode:
        return "save"
    return mode


@pytest.fixture(scope="session")
def commit(request):
    return request.config.getoption("--commit")


@pytest.fixture(scope="session")
def from_commit(request):
    return request.config.getoption("--from")


@pytest.fixture(autouse=True, scope="session")
def data_dir(request, commit, from_commit, tmp_path_factory):
    # read config option to detect "unset" scenario
    mode = request.config.getoption("--mode")
    if mode == "save":
        path = Path(".").absolute() / "_compatibility_data" / commit
        return path
    elif mode == "load":
        path = Path(".").absolute() / "_compatibility_data" / from_commit
        if not path.exists():
            pytest.exit(reason=f"Load path '{path}' does not exist. Please specify a valid load path", returncode=1)
        return path
    # if mode is unset, save and load from a temporary directory
    return Path(tmp_path_factory.mktemp("_compatibility_data"))


@pytest.fixture(params=["sir", "fusion"])
def simulator(request):
    if request.param == "sir":
        from bayesflow.simulators import SIR

        return SIR()
    elif request.param == "fusion":
        from bayesflow.simulators import Simulator
        from bayesflow.types import Shape, Tensor
        from bayesflow.utils.decorators import allow_batch_size
        import numpy as np

        class FusionSimulator(Simulator):
            @allow_batch_size
            def sample(self, batch_shape: Shape, num_observations: int = 4) -> dict[str, Tensor]:
                mean = np.random.normal(0.0, 0.1, size=batch_shape + (2,))
                noise = np.random.standard_normal(batch_shape + (num_observations, 2))

                x = mean[:, None] + noise

                return dict(mean=mean, a=x, b=x)

        return FusionSimulator()
