import keras
import numpy as np
import pytest
from keras.ops import convert_to_numpy as to_np
from keras.ops import convert_to_tensor as to_tensor

from bayesflow.backend import grad, value_and_grad, jvp, vjp, jacrev, jacfwd


def fn_scalar(x):
    return x**2


def fn_multi_scalar(x, y):
    return x * y**2


def fn_vector(x):
    return (x[0] + x[1]) ** 2


def fn_aux(x):
    return x**2, {"info": "aux_data"}


def fn_map(x):
    return x**3


def fn_map_aux(x):
    return x**3, "aux"


@pytest.mark.parametrize("x_val", [3.0, -1.0])
@pytest.mark.parametrize("has_aux", [True, False])
def test_grad_and_value(x_val, has_aux):
    f = fn_aux if has_aux else fn_scalar
    x = to_tensor(x_val)

    # Test grad
    g_fn = grad(f, has_aux=has_aux)
    res = g_fn(x)
    actual_g = to_np(res[0] if has_aux else res)
    np.testing.assert_allclose(actual_g, 2 * x_val, atol=1e-5)

    # Test value_and_grad
    vg_fn = value_and_grad(f, has_aux=has_aux)
    val_aux, dy = vg_fn(x)
    actual_y = to_np(val_aux[0] if has_aux else val_aux)
    actual_dy = to_np(dy)

    np.testing.assert_allclose(actual_y, x_val**2, atol=1e-5)
    np.testing.assert_allclose(actual_dy, 2 * x_val, atol=1e-5)
    if has_aux:
        assert val_aux[1]["info"] == "aux_data"


def test_grad_multi_arg():
    x, y = to_tensor(2.0), to_tensor(3.0)
    # df/dx = y^2 = 9, df/dy = 2xy = 12
    dx, dy = grad(fn_multi_scalar, argnums=(0, 1))(x, y)
    np.testing.assert_allclose(to_np(dx), 9.0, atol=1e-5)
    np.testing.assert_allclose(to_np(dy), 12.0, atol=1e-5)


@pytest.mark.parametrize("has_aux", [True, False])
def test_vjp(has_aux):
    f = fn_aux if has_aux else fn_scalar
    x = to_tensor(3.0)
    v = to_tensor(1.0)

    out, vjp_fn = vjp(f, x, has_aux=has_aux)
    grads = vjp_fn(v)  # Returns tuple of grads

    y = out[0] if has_aux else out
    np.testing.assert_allclose(to_np(y), 9.0, atol=1e-5)
    np.testing.assert_allclose(to_np(grads[0]), 6.0, atol=1e-5)
    if has_aux:
        assert out[1]["info"] == "aux_data"


@pytest.mark.parametrize("has_aux", [True, False])
def test_jvp(has_aux):
    f = fn_map_aux if has_aux else fn_map
    x = to_tensor([2.0, 3.0])
    v = to_tensor([1.0, 0.0])

    # J = [[3x0^2, 0], [0, 3x1^2]] = [[12, 0], [0, 27]]
    # J * v = [12, 0]
    out, tangent = jvp(f, (x,), (v,), has_aux=has_aux)

    y = out[0] if has_aux else out
    np.testing.assert_allclose(to_np(y), [8.0, 27.0], atol=1e-5)
    np.testing.assert_allclose(to_np(tangent), [12.0, 0.0], atol=1e-5)
    if has_aux:
        assert out[1] == "aux"


@pytest.mark.parametrize("jac_fn_name", ["jacrev", "jacfwd"])
@pytest.mark.parametrize("has_aux", [True, False])
def test_jacobians(jac_fn_name, has_aux):
    jac_fn = jacrev if jac_fn_name == "jacrev" else jacfwd
    f = fn_map_aux if has_aux else fn_map
    x = to_tensor([2.0, 4.0])

    # Expected: [[3*2^2, 0], [0, 3*4^2]] = [[12, 0], [0, 48]]
    res = jac_fn(f, has_aux=has_aux)(x)

    j = res[0] if has_aux else res
    np.testing.assert_allclose(to_np(j), [[12.0, 0.0], [0.0, 48.0]], atol=1e-5)
    if has_aux:
        assert res[1] == "aux"


def test_jacobian_multi_arg():
    x = to_tensor([1.0, 2.0])
    y = to_tensor([3.0])

    def f(x, y):
        return keras.ops.sum(x) * y

    # J wrt x: [[y, y]] = [[3, 3]], J wrt y: [[sum(x)]] = [[3]]
    jx, jy = jacrev(f, argnums=(0, 1))(x, y)

    np.testing.assert_allclose(to_np(jx), [[3.0, 3.0]], atol=1e-5)
    np.testing.assert_allclose(to_np(jy), [[3.0]], atol=1e-5)


def test_vjp_multi_arg():
    x, y = to_tensor(2.0), to_tensor(3.0)
    v = to_tensor(1.0)
    # f = x * y^2 = 2 * 9 = 18
    # df/dx = 9, df/dy = 12
    out, vjp_fn = vjp(fn_multi_scalar, x, y)
    grads = vjp_fn(v)

    np.testing.assert_allclose(to_np(out), 18.0, atol=1e-5)
    np.testing.assert_allclose(to_np(grads[0]), 9.0, atol=1e-5)
    np.testing.assert_allclose(to_np(grads[1]), 12.0, atol=1e-5)


# Additional tests for return type consistency
def test_return_types_consistency():
    x = to_tensor(2.0)
    v = to_tensor(1.0)

    # Test grad return types
    g_fn = grad(fn_scalar)
    grad_result = g_fn(x)
    assert hasattr(grad_result, "shape"), "grad should return a tensor-like object"

    g_fn_aux = grad(fn_aux, has_aux=True)
    grad_aux_result = g_fn_aux(x)
    assert isinstance(grad_aux_result, tuple) and len(grad_aux_result) == 2, "grad with aux should return tuple"
    assert hasattr(grad_aux_result[0], "shape"), "first element should be tensor"
    assert isinstance(grad_aux_result[1], dict), "second element should be aux data"

    # Test value_and_grad return types
    vg_fn = value_and_grad(fn_scalar)
    val, grad_val = vg_fn(x)
    assert hasattr(val, "shape"), "value should be tensor"
    assert hasattr(grad_val, "shape"), "grad should be tensor"

    vg_fn_aux = value_and_grad(fn_aux, has_aux=True)
    val_aux, grad_aux = vg_fn_aux(x)
    assert isinstance(val_aux, tuple) and len(val_aux) == 2, (
        "value_and_grad with aux should return ( (val, aux), grad )"
    )
    assert hasattr(val_aux[0], "shape"), "val should be tensor"
    assert isinstance(val_aux[1], dict), "aux should be dict"
    assert hasattr(grad_aux, "shape"), "grad should be tensor"

    # Test vjp return types
    out, vjp_fn = vjp(fn_scalar, x)
    assert hasattr(out, "shape"), "out should be tensor"
    grads = vjp_fn(v)
    assert isinstance(grads, tuple), "vjp_fn should return tuple of grads"
    assert hasattr(grads[0], "shape"), "grad should be tensor"

    out_aux, vjp_fn_aux = vjp(fn_aux, x, has_aux=True)
    assert isinstance(out_aux, tuple) and len(out_aux) == 2, "out with aux should be tuple"
    assert hasattr(out_aux[0], "shape"), "val should be tensor"
    assert isinstance(out_aux[1], dict), "aux should be dict"
    grads_aux = vjp_fn_aux(v)
    assert isinstance(grads_aux, tuple), "grads should be tuple"
    assert hasattr(grads_aux[0], "shape"), "grad should be tensor"

    # Test jvp return types
    primals = (x,)
    tangents = (v,)
    out_jvp, tangent_jvp = jvp(fn_scalar, primals, tangents)
    assert hasattr(out_jvp, "shape"), "out should be tensor"
    assert hasattr(tangent_jvp, "shape"), "tangent should be tensor"

    out_jvp_aux, tangent_jvp_aux = jvp(fn_map_aux, primals, tangents, has_aux=True)
    assert isinstance(out_jvp_aux, tuple) and len(out_jvp_aux) == 2, "out with aux should be tuple"
    assert hasattr(out_jvp_aux[0], "shape"), "val should be tensor"
    assert out_jvp_aux[1] == "aux", "aux should be string"
    assert hasattr(tangent_jvp_aux, "shape"), "tangent should be tensor"

    # Test jacobian return types
    jac_rev = jacrev(fn_map)
    jac_result = jac_rev(x)
    assert hasattr(jac_result, "shape"), "jacobian should be tensor"

    jac_rev_aux = jacrev(fn_map_aux, has_aux=True)
    jac_aux_result = jac_rev_aux(x)
    assert isinstance(jac_aux_result, tuple) and len(jac_aux_result) == 2, "jac with aux should be tuple"
    assert hasattr(jac_aux_result[0], "shape"), "jac should be tensor"
    assert jac_aux_result[1] == "aux", "aux should be string"

    jac_fwd = jacfwd(fn_map)
    jac_fwd_result = jac_fwd(x)
    assert hasattr(jac_fwd_result, "shape"), "jacobian should be tensor"


def test_multi_arg_return_types():
    x, y = to_tensor(2.0), to_tensor(3.0)

    # Test multi-arg grad
    grads_multi = grad(fn_multi_scalar, argnums=(0, 1))(x, y)
    assert isinstance(grads_multi, tuple) and len(grads_multi) == 2, "multi-arg grad should return tuple of grads"
    assert hasattr(grads_multi[0], "shape"), "grad should be tensor"
    assert hasattr(grads_multi[1], "shape"), "grad should be tensor"

    # Test multi-arg vjp
    out_multi, vjp_fn_multi = vjp(fn_multi_scalar, x, y)
    assert hasattr(out_multi, "shape"), "out should be tensor"
    grads_multi_vjp = vjp_fn_multi(to_tensor(1.0))
    assert isinstance(grads_multi_vjp, tuple) and len(grads_multi_vjp) == 2, "multi-arg vjp grads should be tuple"
    assert hasattr(grads_multi_vjp[0], "shape"), "grad should be tensor"
    assert hasattr(grads_multi_vjp[1], "shape"), "grad should be tensor"

    # Test multi-arg jacobian
    jac_multi = jacrev(fn_multi_scalar, argnums=(0, 1))(x, y)
    assert isinstance(jac_multi, tuple) and len(jac_multi) == 2, "multi-arg jac should return tuple of jacs"
    assert hasattr(jac_multi[0], "shape"), "jac should be tensor"
    assert hasattr(jac_multi[1], "shape"), "jac should be tensor"
