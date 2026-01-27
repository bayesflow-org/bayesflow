import keras

from bayesflow.types import Tensor


def squared_euclidean(x1: Tensor, x2: Tensor) -> Tensor:
    # flatten trailing dims
    x1 = keras.ops.reshape(x1, (keras.ops.shape(x1)[0], -1))
    x2 = keras.ops.reshape(x2, (keras.ops.shape(x2)[0], -1))

    x1_sq = keras.ops.sum(x1 * x1, axis=1, keepdims=True)  # (n,1)
    x2_sq = keras.ops.sum(x2 * x2, axis=1, keepdims=True)  # (m,1)
    cross = keras.ops.matmul(x1, keras.ops.transpose(x2))  # (n,m)

    dist2 = x1_sq + keras.ops.transpose(x2_sq) - 2.0 * cross
    return keras.ops.maximum(dist2, 0.0)


def cosine_distance(x1: Tensor, x2: Tensor, eps: float = 1e-8) -> Tensor:
    """
    Pairwise cosine distance:
        d(x, y) = 1 - <x, y> / (||x|| ||y||)

    x1: Tensor of shape (n, ...)
    x2: Tensor of shape (m, ...)
    returns: Tensor of shape (n, m)
    """
    x1 = keras.ops.reshape(x1, (keras.ops.shape(x1)[0], -1))
    x2 = keras.ops.reshape(x2, (keras.ops.shape(x2)[0], -1))

    x1 = x1 / (keras.ops.norm(x1, axis=1, keepdims=True) + eps)
    x2 = x2 / (keras.ops.norm(x2, axis=1, keepdims=True) + eps)

    # cosine similarity
    sim = keras.ops.matmul(x1, keras.ops.transpose(x2))
    sim = keras.ops.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def augment_for_partial_ot(
    cost_scaled: Tensor,
    regularization: float,
    s: float,
    dummy_cost: float | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Augments a scaled cost matrix for partial OT via a dummy row/column.

    For partial OT with mass s ∈ (0,1), we transport s proportion of mass
    and leave (1-s) unmatched via dummy nodes.
    """
    if dummy_cost is None:
        dummy_cost = keras.ops.max(-cost_scaled * regularization) + 1  # same as POT library default

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
    M: Tensor,
    C: Tensor,
    condition_ratio: float,
    initial_w: float = 1.0,
    max_iter: int = 10,
    abs_tol: float = 1e-3,
    max_w: float = 1e8,
) -> tuple[Tensor, Tensor]:
    """
    Find w such that mean((M + w*C) <= diag(M)) ≈ condition_ratio

    Returns:
      cost = M + w*C   (Tensor, shape (N,N))
      w    = Tensor scalar (same dtype as M)
    """
    dtype = M.dtype
    r_t = keras.ops.convert_to_tensor(condition_ratio, dtype=dtype)
    max_w_t = keras.ops.convert_to_tensor(max_w, dtype=dtype)

    # condition: M + w*C <= M_diag  =>  w*C <= M_diag - M
    M_diag = keras.ops.expand_dims(keras.ops.diagonal(M), 1)
    Delta = M_diag - M  # Pre-computed target threshold

    def get_ratio(w):
        return keras.ops.mean(keras.ops.cast(w * C <= Delta, dtype))

    # Boundary check at w=0
    r0 = get_ratio(keras.ops.convert_to_tensor(0.0, dtype=dtype))

    def do_search():
        # Exponential search to bracket w
        def exp_cond(it, low, high):
            return (it < max_iter) & (get_ratio(high) > r_t) & (high < max_w_t)

        def exp_body(it, low, high):
            return it + 1, high, high * 2.0

        _, low, high = keras.ops.while_loop(exp_cond, exp_body, (0, 0.0, initial_w))
        high = keras.ops.minimum(high, max_w_t)

        # Binary search for optimal w
        def bin_cond(it, low, high, best_w, best_r):
            return (it < max_iter) & (keras.ops.abs(best_r - r_t) > abs_tol)

        def bin_body(it, low, high, best_w, best_r):
            mid = (low + high) / 2.0
            r_mid = get_ratio(mid)

            # Update bounds
            new_high = keras.ops.where(r_mid < r_t, mid, high)
            new_low = keras.ops.where(r_mid < r_t, low, mid)

            # Update best_w based on proximity to target ratio
            closer = keras.ops.abs(r_mid - r_t) < keras.ops.abs(best_r - r_t)
            best_w_next = keras.ops.where(closer, mid, best_w)
            best_r_next = keras.ops.where(closer, r_mid, best_r)

            return it + 1, new_low, new_high, best_w_next, best_r_next

        _, _, _, final_w, _ = keras.ops.while_loop(bin_cond, bin_body, (0, low, high, high, get_ratio(high)))
        return final_w

    # Select w and construct final cost matrix once
    optimal_w = keras.ops.cond(r0 < r_t, lambda: 0.0, do_search)
    final_cost = M + optimal_w * C

    return final_cost, optimal_w
