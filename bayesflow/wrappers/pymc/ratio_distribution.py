from collections.abc import Callable, Sequence

import keras

import pymc as pm
import pytensor
import pytensor.tensor as pt

from .backend import Ratio


class RatioDistribution:
    """
    User-facing wrapper around :class:`pm.CustomDist` for ratio-based likelihoods.

    Adapts a trained :class:`~bayesflow.approximators.RatioApproximator` for use
    as a likelihood term inside a PyMC model.  The approximated log-ratio
    ``log p(x|θ) / p(x)`` is used as an unnormalised log-likelihood, which is
    valid for posterior inference because the evidence ``p(x)`` is constant
    with respect to the parameters.

    Parameters
    ----------
    ratio_approximator : ~bayesflow.approximators.RatioApproximator
        A trained ratio approximator whose ``log_ratio`` method will be called
        to evaluate the likelihood.
    param_names : sequence of str
        Names of the model parameters in the order expected by the ratio
        approximator.  The same order must be used when registering this
        distribution in a PyMC model (see :meth:`__call__`).
    exchangeable : bool, optional
        Controls how the log-ratio is aggregated over observations.

        ``True`` (default) — i.i.d. / exchangeable likelihood:
            The ratio network is evaluated independently for each
            observation and the results are summed.  Parameters are
            broadcast to match the number of trials, so they may be
            either scalars (shared across trials) or per-trial vectors.
            PyMC ``signature``: ``"(),()->()"``.  The Op returns a
            per-trial vector; PyMC takes care of the summation.

        ``False`` — joint / non-exchangeable likelihood:
            The ratio network receives the full dataset at once and
            returns a single scalar log-ratio.  No vectorisation is
            applied.  PyMC ``signature``: ``"(n),()->()"``.  The Op
            returns a scalar directly.
    simulator_fn : callable or None, optional
        A function with signature ``(rng, *params, size) -> ndarray`` used
        to generate samples.  Required only when prior- or posterior-predictive
        sampling is needed (e.g. :func:`pm.sample_prior_predictive`).

    Examples
    --------
    >>> import pymc as pm
    >>> from bayesflow.wrappers.pymc import RatioDistribution
    >>> ratio_dist = RatioDistribution(
    ...     ratio_approximator=approximator,
    ...     param_names=["mu", "sigma"],
    ... )
    >>> with pm.Model():
    ...     mu = pm.Normal("mu", mu=0, sigma=1)
    ...     sigma = pm.HalfNormal("sigma", sigma=1)
    ...     obs = ratio_dist("obs", mu=mu, sigma=sigma, observed=data)
    ...     trace = pm.sample()
    """

    def __init__(
        self,
        ratio_approximator,
        param_names: Sequence[str],
        *,
        exchangeable: bool = True,
        simulator_fn: Callable | None = None,
    ):
        self.param_names = tuple(param_names)
        self.exchangeable = exchangeable
        self.simulator_fn = simulator_fn
        self.backend = Ratio(ratio_approximator, param_names, exchangeable=exchangeable)

    @property
    def log_ratio(self) -> Callable:
        return self.backend.log_ratio

    def _prepare_params(self, args, kwargs):
        if args and kwargs:
            raise TypeError("Pass parameters either positionally or by keyword, not both.")

        if args:
            if len(args) != len(self.param_names):
                raise TypeError(f"Expected {len(self.param_names)} parameters, got {len(args)}.")
            return list(args)

        missing = [name for name in self.param_names if name not in kwargs]
        if missing:
            raise TypeError(f"Missing required parameters: {missing}")

        extra = [name for name in kwargs if name not in self.param_names]
        if extra:
            raise TypeError(f"Unexpected parameters: {extra}")

        return [kwargs[name] for name in self.param_names]

    def logp(self, value: pt.TensorVariable, *dist_params: pt.TensorVariable) -> pt.TensorVariable:
        """
        Symbolic log-probability for ``pm.CustomDist``.

        Parameters
        ----------
        value : pt.TensorVariable
            Observed data tensor passed in by PyMC.
        *dist_params : pt.TensorVariable
            Distribution parameters in the order given by ``param_names``.

        Returns
        -------
        pt.TensorVariable
            A vector of per-trial log-ratios (exchangeable mode) or a
            scalar joint log-ratio (non-exchangeable mode).

        Notes
        -----
        *Exchangeable mode* — PyMC may pass scalar parameters as tensors
        with shape ``(1,)`` instead of true 0-d scalars.  Every parameter
        is flattened and broadcast to the flattened data shape so JAX
        ``vmap`` receives matching ``(n_obs,)`` vectors.

        *Non-exchangeable mode* — The full data vector is forwarded
        as-is; parameters remain scalars and the Op returns a single
        scalar log-ratio.
        """
        value = pt.cast(pt.as_tensor_variable(value), pytensor.config.floatX)

        if self.exchangeable:
            value = pt.reshape(value, (-1,))  # (n_obs,)
            prepared_params = []
            for p in dist_params:
                p = pt.cast(pt.as_tensor_variable(p), pytensor.config.floatX)
                p = pt.reshape(p, (-1,))  # scalar -> (1,), vector -> (n,)
                p = pt.broadcast_to(p, value.shape)
                prepared_params.append(p)
        else:
            prepared_params = [pt.cast(pt.as_tensor_variable(p), pytensor.config.floatX) for p in dist_params]

        return self.backend.logp_op(value, *prepared_params)

    def random(self, *dist_params, rng=None, size=None):
        """
        Draw random samples from the model (requires ``simulator_fn``).

        Parameters
        ----------
        *dist_params :
            Distribution parameters in the order given by ``param_names``.
        rng : numpy.random.Generator or None, optional
            Random number generator passed by PyMC.
        size : tuple[int, ...] or None, optional
            Shape of the requested sample.

        Returns
        -------
        numpy.ndarray
            Simulated observations.

        Raises
        ------
        NotImplementedError
            If no ``simulator_fn`` was provided at construction time.
        """
        if self.simulator_fn is None:
            raise NotImplementedError("No simulator_fn was provided, so random sampling is unavailable.")
        return self.simulator_fn(rng, *dist_params, size=size)

    def __call__(self, name: str, *args, observed=None, **kwargs) -> pm.Distribution:
        """
        Register this distribution as a named variable in the active PyMC model.

        Parameters
        ----------
        name : str
            Name of the PyMC random variable.
        *args :
            Distribution parameters passed positionally, in the order
            given by ``param_names``.
        observed : array-like or None, optional
            Observed data to condition on.
        **kwargs :
            Distribution parameters passed by keyword.  Cannot be mixed
            with positional ``args``.

        Returns
        -------
        pm.Distribution
            A ``pm.CustomDist`` node wired into the active PyMC model.
        """
        params = self._prepare_params(args, kwargs)

        if self.exchangeable:
            # Each element of `observed` is an independent scalar draw
            signature = ",".join(["()"] * len(self.param_names)) + "->()"
        else:
            # The whole `observed` array is a single joint observation
            signature = "(n)," + ",".join(["()"] * len(self.param_names)) + "->()"

        custom_kwargs = {
            "logp": self.logp,
            "signature": signature,
            "dtype": keras.backend.floatx(),
        }

        if self.simulator_fn is not None:
            custom_kwargs["random"] = self.random

        return pm.CustomDist(
            name,
            *params,
            observed=observed,
            **custom_kwargs,
        )
