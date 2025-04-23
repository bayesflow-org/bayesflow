import keras

from .log_sinkhorn import log_sinkhorn
from .sinkhorn import sinkhorn

methods = {
    "sinkhorn": sinkhorn,
    "sinkhorn_knopp": sinkhorn,
    "log_sinkhorn": log_sinkhorn,
    "log_sinkhorn_knopp": log_sinkhorn,
}


def optimal_transport(x1, x2, method="log_sinkhorn", return_assignments=False, **kwargs):
    """Matches elements from x2 onto x1, such that the transport cost between them is minimized, according to the method
    and cost matrix used.

    Depending on the method used, elements in either tensor may be permuted, dropped, duplicated, or otherwise modified,
    such that the assignment is optimal.

    Note: this is just a dispatch function that calls the appropriate optimal transport method.
    See the documentation of the respective method for more details.

    :param x1: Tensor of shape (n, ...)
        Samples from the first distribution.

    :param x2: Tensor of shape (m, ...)
        Samples from the second distribution.

    :param method: Method used to compute the transport cost.
        Default: 'log_sinkhorn'

    :param return_assignments: Whether to return the assignment indices.
        Default: False

    :param kwargs: Additional keyword arguments that are passed to the optimization method.

    :return: Tensors of shapes (n, ...) and (m, ...)
        x1 and x2 in optimal transport permutation order.
    """
    assignments = methods[method.lower()](x1, x2, **kwargs)
    x2 = keras.ops.take(x2, assignments, axis=0)

    if return_assignments:
        return x1, x2, assignments

    return x1, x2
