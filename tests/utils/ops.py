import contextlib

import keras
import numpy as np


def on_torch_mps():
    """Whether keras places tensors on the MPS (Apple GPU) device of the torch backend.

    Useful to guard tests that fail because of MPS limitations, e.g. its lack of float64 support.
    """
    if keras.backend.backend() != "torch":
        return False

    return keras.ops.convert_to_tensor(0.0).device.type == "mps"


def cpu_if_torch_mps():
    """Context manager that runs the enclosed block on the CPU if keras would use torch's MPS device."""
    if on_torch_mps():
        return keras.device("cpu")

    return contextlib.nullcontext()


def allclose(x1, x2, rtol=1e-5, atol=1e-5):
    return keras.ops.all(keras.ops.isclose(x1, x2, rtol, atol))


def assert_allclose(x1, x2, rtol=1e-5, atol=1e-8, msg=""):
    x1 = keras.ops.convert_to_numpy(x1)
    x2 = keras.ops.convert_to_numpy(x2)

    assert x1.shape == x2.shape, "Input shapes do not match."

    mse = np.mean(np.square(x1 - x2)).item()
    largest_deviation = np.max(np.abs(x1 - x2)).item()
    largest_deviation_index = np.unravel_index(np.argmax(np.abs(x1 - x2)), x1.shape)
    largest_deviation_value1 = x1[largest_deviation_index].item()
    largest_deviation_value2 = x2[largest_deviation_index].item()

    if msg:
        msg = f"{msg}\n"
    else:
        msg = "Inputs significantly differ:\n"

    msg += "Largest Deviation:\n"
    msg += f"|{largest_deviation_value1:.02e} - {largest_deviation_value2:.02e}| = {largest_deviation:.02e}\n"
    msg += "\n"
    msg += "MSE:\n"
    msg += f"{mse:.02e}"

    assert allclose(x1, x2, rtol, atol), msg


def max_mean_discrepancy(x, y):
    # Computes the Max Mean Discrepancy between samples of two distributions
    xx = keras.ops.matmul(x, keras.ops.transpose(x))
    yy = keras.ops.matmul(y, keras.ops.transpose(y))
    zz = keras.ops.matmul(x, keras.ops.transpose(y))

    rx = keras.ops.broadcast_to(keras.ops.expand_dims(keras.ops.diag(xx), 0), xx.shape)
    ry = keras.ops.broadcast_to(keras.ops.expand_dims(keras.ops.diag(yy), 0), yy.shape)

    dxx = keras.ops.transpose(rx) + rx - 2.0 * xx
    dyy = keras.ops.transpose(ry) + ry - 2.0 * yy
    dxy = keras.ops.transpose(rx) + ry - 2.0 * zz

    XX = keras.ops.zeros(xx.shape)
    YY = keras.ops.zeros(yy.shape)
    XY = keras.ops.zeros(zz.shape)

    # RBF scaling
    bandwidth = [10, 15, 20, 50]
    for a in bandwidth:
        XX += keras.ops.exp(-0.5 * dxx / a)
        YY += keras.ops.exp(-0.5 * dyy / a)
        XY += keras.ops.exp(-0.5 * dxy / a)

    return keras.ops.mean(XX + YY - 2.0 * XY)
