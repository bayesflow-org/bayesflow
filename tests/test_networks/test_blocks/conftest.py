import pytest

from bayesflow.networks.helpers import DenseBlock, ConditionalDenseBlock, DiffusionTransformerBlock


@pytest.fixture()
def dense_block():
    return DenseBlock(width=8)


@pytest.fixture()
def dense_build_shapes():
    return {"input_shape": (4, 6)}


@pytest.fixture()
def cond_build_shapes():
    return {"input_shape": ((4, 6), (4, 3))}


@pytest.fixture(params=["conditional_dense", "diffusion_transformer"])
def conditioning_block(request):
    """A block taking a ``(x, conditioning)`` tuple, bundled with matching shapes.

    ``make(**kwargs)`` returns a fresh block with the given overrides so tests can
    tweak e.g. ``dropout`` without sharing build state across cases.
    """
    if request.param == "conditional_dense":
        return {
            "make": lambda **kwargs: ConditionalDenseBlock(width=8, **kwargs),
            "build_shapes": {"input_shape": ((4, 6), (4, 3))},
            "x_shape": (4, 6),
            "cond_shape": (4, 3),
            "out_shape": (4, 8),
        }
    return {
        "make": lambda **kwargs: DiffusionTransformerBlock(width=8, num_heads=4, **kwargs),
        "build_shapes": {"input_shape": ((4, 5, 8), (4, 48))},
        "x_shape": (4, 5, 8),
        "cond_shape": (4, 48),
        "out_shape": (4, 5, 8),
    }
