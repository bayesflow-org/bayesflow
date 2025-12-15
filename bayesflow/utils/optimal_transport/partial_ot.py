import keras
from bayesflow.types import Tensor


def augment_for_partial_ot(
    cost_scaled: Tensor,
    regularization: float,
    s: float,
    dummy_cost: float,
) -> tuple[Tensor, Tensor, Tensor]:
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


def search_for_conditional_weight(
    M,  # (N,N) nonnegative data cost
    C,  # (N,N) nonnegative conditional cost
    condition_ratio: float,  # target ratio
    initial_w: float = 1.0,
    max_iter: int = 10,
    abs_tol: float = 1e-3,
    max_w: float = 1e8,
):
    """
    Find w such that mean((M + w*C) <= diag(M)) ≈ condition_ratio

    Returns:
      cost = M + w*C   (Tensor, shape (N,N))
      w    = Tensor scalar (same dtype as M)
    """
    dtype = M.dtype
    r_t = keras.ops.convert_to_tensor(condition_ratio, dtype=dtype)
    abs_tol_t = keras.ops.convert_to_tensor(abs_tol, dtype=dtype)
    max_w_t = keras.ops.convert_to_tensor(max_w, dtype=dtype)

    # diag(M) as (N,1) for broadcasting
    n = keras.ops.shape(M)[0]
    idx = keras.ops.arange(n)
    oh = keras.ops.one_hot(idx, n, dtype=dtype)
    M_diag = keras.ops.sum(M * oh, axis=1)  # (N,)
    M_diag = keras.ops.reshape(M_diag, (-1, 1))  # (N,1)

    def r_fn(w):
        cost = M + w * C
        curr_r = keras.ops.mean(keras.ops.cast(cost <= M_diag, dtype))
        return cost, curr_r

    # r is maximized at w=0; if already below target, return w=0
    cost0, r0 = r_fn(keras.ops.convert_to_tensor(0.0, dtype=dtype))

    def return_zero():
        return cost0, keras.ops.convert_to_tensor(0.0, dtype=dtype)

    def do_search():
        low0 = keras.ops.convert_to_tensor(0.0, dtype=dtype)
        high0 = keras.ops.convert_to_tensor(initial_w, dtype=dtype)

        # ---- exponential search to bracket ----
        def exp_cond(it, low, high):
            _, curr_r = r_fn(high)
            return (it < max_iter) & (curr_r > r_t) & (high <= max_w_t)

        def exp_body(it, low, high):
            low = high
            high = high * 2.0
            return it + 1, low, high

        it0 = keras.ops.convert_to_tensor(0, dtype="int32")
        it, low, high = keras.ops.while_loop(exp_cond, exp_body, (it0, low0, high0))
        high = keras.ops.minimum(high, max_w_t)

        # ---- binary search ----
        # Keep best (w,cost,r) encountered to return even if we don't hit tol.
        best_w0 = high
        best_cost0, best_r0 = r_fn(high)

        def bin_cond(it, low, high, best_w, best_cost, best_r):
            return (it < max_iter) & (keras.ops.abs(best_r - r_t) > abs_tol_t)

        def bin_body(it, low, high, best_w, best_cost, best_r):
            mid = (low + high) / 2.0
            cost_mid, r_mid = r_fn(mid)

            # if r_mid < r: w too large => move high down, else move low up
            high = keras.ops.where(r_mid < r_t, mid, high)
            low = keras.ops.where(r_mid < r_t, low, mid)

            # update best if closer to target
            better = keras.ops.abs(r_mid - r_t) < keras.ops.abs(best_r - r_t)
            best_w = keras.ops.where(better, mid, best_w)
            best_r = keras.ops.where(better, r_mid, best_r)
            best_cost = keras.ops.where(better, cost_mid, best_cost)

            return it + 1, low, high, best_w, best_cost, best_r

        it1 = keras.ops.convert_to_tensor(0, dtype="int32")
        it, low, high, best_w, best_cost, best_r = keras.ops.while_loop(
            bin_cond, bin_body, (it1, low, high, best_w0, best_cost0, best_r0)
        )
        return best_cost, best_w

    return keras.ops.cond(r0 < r_t, return_zero, do_search)
