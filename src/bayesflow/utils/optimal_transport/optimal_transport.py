import keras

from bayesflow.types import Tensor

from .log_sinkhorn import log_sinkhorn
from .sinkhorn import sinkhorn

methods = {
    "sinkhorn": sinkhorn,
    "sinkhorn_knopp": sinkhorn,
    "log_sinkhorn": log_sinkhorn,
    "log_sinkhorn_knopp": log_sinkhorn,
}


def optimal_transport(
    x1: Tensor, x2: Tensor, conditions: Tensor | None = None, method="sinkhorn", return_assignments=False, **kwargs
) -> tuple[Tensor, Tensor, Tensor | None, Tensor] | tuple[Tensor, Tensor, Tensor | None]:
    """
    Match elements from ``x2`` onto ``x1`` by minimizing the transport cost.

    This function dispatches to a specific optimal transport method according to
    the selected ``method`` and cost formulation. Depending on the method used,
    elements in either tensor may be permuted, dropped, duplicated, or otherwise
    modified in order to achieve an optimal assignment.

    Note
    ----
    This is a dispatch function that calls the appropriate optimal transport
    implementation. See the documentation of the selected method for details on
    the exact optimization procedure and assumptions.

    Parameters
    ----------
    x1 : Tensor
        Tensor of shape ``(n, ...)`` containing samples from the first distribution.
    x2 : Tensor
        Tensor of shape ``(m, ...)`` containing samples from the second distribution.
    conditions : Tensor, optional
        Tensor of shape ``(k, ...)`` providing conditioning information for
        conditional optimal transport. If ``None``, unconditional optimal transport
        is performed. Default is ``None``.
    method : str, optional
        Method used to compute the optimal transport plan (e.g., ``'sinkhorn'``).
        Default is ``'sinkhorn'``.
    return_assignments : bool
        If ``True``, also return the assignment indices produced by the transport
        method. Default is ``False``.
    **kwargs
        Additional keyword arguments passed to the selected optimal transport method.

    Returns
    -------
    Tuple of tensors
        If ``return_assignments`` is ``False``, returns three tensors of shapes
        ``(n, ...)`` and ``(m, ...)`` corresponding to ``x1``, ``x2``, ``conditions`` reordered
        according to the optimal transport solution. If ``return_assignments`` is
        ``True``, the reordered tensors and the corresponding assignment indices
        are returned as a fourth element.
    """

    assignments = methods[method.lower()](x1, x2, conditions, **kwargs)
    x2 = keras.ops.take(x2, assignments, axis=0)

    if conditions is not None:
        # conditions must be resampled along with x1
        conditions = keras.ops.take(conditions, assignments, axis=0)

    if return_assignments:
        return x1, x2, conditions, assignments

    return x1, x2, conditions
