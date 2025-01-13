import keras


def isclose(x1, x2, rtol=1e-5, atol=1e-8):
    return keras.ops.abs(x1 - x2) <= atol + rtol * keras.ops.abs(x2)


def allclose(x1, x2, rtol=1e-5, atol=1e-8):
    return keras.ops.all(isclose(x1, x2, rtol, atol))


def assert_allclose(x1, x2, rtol=1e-5, atol=1e-8, msg=""):
    mse = keras.ops.mean(keras.ops.square(x1 - x2))
    assert allclose(x1, x2, rtol, atol), f"{msg} - mse={mse}"


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
