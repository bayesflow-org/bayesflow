import keras
from bayesflow.types import Tensor


def augment_for_partial_ot(
    cost_scaled: Tensor,
    regularization: float,
    s: float,
    dummy_cost: float,
):
    """
    Augments a scaled cost matrix for partial OT via a dummy row/column and returns
    (cost_scaled_aug, a, b, real_slice), where:
      - cost_scaled_aug has shape (n+1, m+1)
      - a is row-marginal vector of shape (n+1,)
      - b is col-marginal vector of shape (m+1,)
      - real_slice is a tuple of slices to extract the real-to-real block.
    """
    # cost_scaled is expected to be -C/eps with shape (n, m)
    A = keras.ops.convert_to_tensor(dummy_cost, dtype=cost_scaled.dtype)

    n0 = keras.ops.shape(cost_scaled)[0]
    m0 = keras.ops.shape(cost_scaled)[1]

    # Augmented cost: [[cost_scaled, 0], [0, -A/eps]]
    zero_col = keras.ops.zeros((n0, 1), dtype=cost_scaled.dtype)  # (n0, 1)
    zero_row = keras.ops.zeros((1, m0), dtype=cost_scaled.dtype)  # (1, m0)
    br = keras.ops.reshape(-A / regularization, (1, 1))  # (1, 1)

    top = keras.ops.concatenate([cost_scaled, zero_col], axis=1)  # (n0, m0+1)
    bottom = keras.ops.concatenate([zero_row, br], axis=1)  # (1, m0+1)
    cost_scaled_aug = keras.ops.concatenate([top, bottom], axis=0)  # (n0+1, m0+1)

    # Augmented marginals: [u_n, 1-s] and [u_m, 1-s]
    dtype = cost_scaled.dtype
    s_t = keras.ops.convert_to_tensor(s, dtype=dtype)
    one_minus_s = 1.0 - s_t

    n0_f = keras.ops.cast(n0, dtype)
    m0_f = keras.ops.cast(m0, dtype)

    a = keras.ops.concatenate(
        [
            keras.ops.ones((n0,), dtype=dtype) * (1.0 / n0_f),
            keras.ops.reshape(one_minus_s, (1,)),
        ],
        axis=0,
    )  # (n0+1,)

    b = keras.ops.concatenate(
        [
            keras.ops.ones((m0,), dtype=dtype) * (1.0 / m0_f),
            keras.ops.reshape(one_minus_s, (1,)),
        ],
        axis=0,
    )  # (m0+1,)

    return cost_scaled_aug, a, b
