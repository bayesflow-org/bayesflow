import numpy as np
import keras
import pytest

from bayesflow.utils.serialization import deserialize, serialize
from tests.utils import assert_allclose


@pytest.fixture()
def matrix_adapter():
    from bayesflow.adapters import Adapter

    return (
        Adapter(differentiable=True)
        .as_covariance_matrix("cov")
        .as_covariance_matrix("cov_chol", cholesky=True)
        .as_correlation_matrix("cor")
        .as_correlation_matrix("cor_chol", cholesky=True)
    )


@pytest.fixture()
def matrix_data():
    samples = np.random.standard_normal(size=(32, 10, 4))

    cov = np.stack([np.cov(s, rowvar=False) for s in samples])
    cov_chol = np.linalg.cholesky(cov)
    cor = np.stack([np.corrcoef(s, rowvar=False) for s in samples])
    cor_chol = np.linalg.cholesky(cor)

    return {"cov": cov, "cov_chol": cov_chol, "cor": cor, "cor_chol": cor_chol}


@pytest.mark.cpu_fallback_on_mps
def test_cycle_consistency(matrix_adapter, matrix_data):
    transformed = matrix_adapter(matrix_data)
    restored = matrix_adapter(transformed, inverse=True)

    for key, value in matrix_data.items():
        batch_size, k, _ = value.shape

        # compare to expected shapes
        dim = k * (k - 1) // 2
        if key.startswith("cov"):
            dim += k
        assert keras.ops.shape(transformed[key]) == (batch_size, dim)

        assert_allclose(restored[key], value)

    # matrix and its cholesky factor should map to the same unconstrained values
    for key in ["cov", "cor"]:
        assert_allclose(transformed[key], transformed[f"{key}_chol"])


@pytest.mark.cpu_fallback_on_mps
def test_serialize_deserialize(matrix_adapter, matrix_data):
    processed = matrix_adapter(matrix_data)
    deserialized = deserialize(serialize(matrix_adapter))
    deserialized_processed = deserialized(matrix_data)

    for key, value in processed.items():
        assert_allclose(deserialized_processed[key], value)


@pytest.mark.jax
def test_log_det_jac(matrix_adapter, matrix_data):
    # tests against JAX autodiff
    import jax

    transformed, log_det_jac = matrix_adapter(matrix_data, log_det_jac=True)

    for key, value in transformed.items():
        # free entries of the (cholesky factor of the) matrix: the lower triangle
        k = matrix_data[key].shape[-1]
        rows, cols = np.tril_indices(k, k=0 if key.startswith("cov") else -1)

        # vmap removes the batch dim: add it with y[None] and remove it with [0]
        def inverse(y):
            return matrix_adapter({key: y[None]}, inverse=True)[key][0][rows, cols]

        jacobian = jax.vmap(jax.jacobian(inverse))(value)
        expected = -np.linalg.slogdet(jacobian)[1]
        assert_allclose(log_det_jac[key], expected, rtol=1e-3, atol=1e-3)
