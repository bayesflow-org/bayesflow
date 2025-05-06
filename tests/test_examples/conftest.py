import pytest


@pytest.fixture(scope="session")
def examples_path():
    from pathlib import Path

    return Path(__file__).parents[2] / "examples"
