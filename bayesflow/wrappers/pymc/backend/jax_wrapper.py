from collections.abc import Callable, Sequence
from abc import abstractmethod

from jax import jit, vjp, vmap

from ..pytensor_ops import RatioLogpOp, RatioLogpVJPOp


class JAXWrapper:
    """
    Abstract base for JAX-backed PyTensor log-probability Ops.

    Subclasses implement only :meth:`make_log_prob`, which returns a callable
    ``(x, *params) -> scalar | array``.  All ``vmap`` / ``jit`` / ``vjp``
    wiring and PyTensor Op construction is handled here.

    Parameters
    ----------
    approximator :
        A trained BayesFlow approximator.
    param_names :
        Names of the model parameters in the order they will be passed
        at inference time.
    exchangeable : bool, optional
        If ``True`` (default) the log-prob callable is vmapped over the
        first axis of the data, treating observations as i.i.d.; the Op
        returns a per-observation vector that PyMC sums.
        If ``False`` the callable receives the full data array at once
        and returns a scalar.
    """

    def __init__(
        self,
        approximator,
        param_names: Sequence[str],
        *,
        exchangeable: bool = True,
    ):
        self.approximator = approximator
        self.param_names = tuple(param_names)
        self.exchangeable = exchangeable

        self.log_prob = self.make_log_prob()
        self.logp_nojit, self.logp_jit, self.logp_vjp_nojit, self.logp_vjp_jit = self.make_logp_functions()

        self.logp_vjp_op = RatioLogpVJPOp(self.logp_vjp_jit, self.logp_vjp_nojit)
        self.logp_op = RatioLogpOp(
            logp_jit=self.logp_jit,
            logp_nojit=self.logp_nojit,
            vjp_op=self.logp_vjp_op,
            scalar_output=not exchangeable,
        )

    @abstractmethod
    def make_log_prob(self) -> Callable:
        """
        Return a callable ``(x, *params) -> scalar | array``.

        With ``exchangeable=True`` this is called per-observation via
        ``vmap``; with ``exchangeable=False`` it receives the full data
        array.  Called once during ``__init__``; stored as ``self.log_prob``
        and consumed by :meth:`make_logp_functions`.
        """
        raise NotImplementedError

    def make_logp_functions(self) -> tuple[Callable, Callable, Callable, Callable]:
        """
        Build the JIT-compiled log-prob and its VJP from ``self.log_prob``.

        Returns
        -------
        logp_nojit : Callable
            ``(data, *params) -> array`` — optionally vmapped, no JIT.
        logp_jit : Callable
            JIT-compiled version of ``logp_nojit``.
        logp_vjp_nojit : Callable
            ``(data, *params, gz) -> tuple[array, ...]`` — VJP w.r.t.
            params, no JIT.  Used by JAX-backend samplers via
            ``jax_funcify``.
        logp_vjp_jit : Callable
            JIT-compiled version of ``logp_vjp_nojit``.  Used by the
            pytensor C/Python ``perform()`` path.
        """
        n_params = len(self.param_names)

        if self.exchangeable:
            logp_nojit = vmap(self.log_prob, in_axes=(0,) + (0,) * n_params)
        else:
            logp_nojit = self.log_prob

        logp_jit = jit(logp_nojit)

        def logp_vjp_nojit(data, *params, gz):
            _, vjp_fn = vjp(logp_nojit, data, *params)
            return vjp_fn(gz)[1:]  # drop grad w.r.t. observed data

        logp_vjp_jit = jit(logp_vjp_nojit)

        return logp_nojit, logp_jit, logp_vjp_nojit, logp_vjp_jit
