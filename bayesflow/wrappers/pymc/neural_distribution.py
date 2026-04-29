from collections.abc import Callable, Sequence
from numpy.typing import ArrayLike

import numpy as np

import keras

import pymc as pm
import pytensor
import pytensor.tensor as pt

from .backend import build_backend


class NeuralDistribution:
    """
    Wraps a trained BayesFlow approximator as a PyMC custom distribution.

    Supports two approximator types:

    * :class:`~bayesflow.approximators.RatioApproximator` — Neural Ratio Estimation
      (NRE).  The log-ratio ``log p(x|θ)/p(x)`` is used as an unnormalised
      log-likelihood (the evidence is constant w.r.t. θ and cancels in the
      posterior).

    * :class:`~bayesflow.approximators.ContinuousApproximator` — Neural Likelihood
      Estimation (NLE).  The network must be trained in **NLE mode**: observations
      as ``inference_variables``, parameters as ``inference_conditions``.  The
      network models ``log p(x|θ)`` directly.

    In both cases the JAX backend vmaps the per-trial function over the trial axis
    and registers an exact VJP so gradient-based samplers (NUTS, HMC) work without
    finite differences.

    Parameters
    ----------
    approximator :
        A trained ``RatioApproximator`` or ``ContinuousApproximator``.
    param_names : sequence of str
        Names of the model parameters in the order they will be passed inside
        the PyMC model.
    exchangeable : bool, optional
        If ``True`` (default) observations are treated as i.i.d.: the network
        is vmapped over trials and returns a per-trial log-value vector which
        PyMC sums.  If ``False`` the full dataset is forwarded to the network
        at once and a scalar log-value is returned.
    simulator_fn : callable or None, optional
        A function ``(rng, *params, size) -> ndarray`` for prior/posterior
        predictive sampling.  Required only when
        :func:`pm.sample_prior_predictive` or similar is used.

    Examples
    --------
    **Ratio estimator (NRE)**::

        dist = NeuralDistribution(ratio_approximator, param_names=["mu", "sigma"])
        with pm.Model():
            mu = pm.Normal("mu", 0, 1)
            sigma = pm.HalfNormal("sigma", 1)
            obs = dist("obs", mu=mu, sigma=sigma, observed=data)

    **Likelihood estimator (NLE)**::

        dist = NeuralDistribution(nle_approximator, param_names=["mu", "sigma"])
        with pm.Model():
            mu = pm.Normal("mu", 0, 1)
            sigma = pm.HalfNormal("sigma", 1)
            obs = dist("obs", mu=mu, sigma=sigma, observed=data)
    """

    def __init__(
        self,
        approximator,
        param_names: Sequence[str],
        *,
        exchangeable: bool = True,
        simulator_fn: Callable | None = None,
    ):
        self.param_names = tuple(param_names)
        self.exchangeable = exchangeable
        self.simulator_fn = simulator_fn
        self.backend = build_backend(approximator, param_names, exchangeable=exchangeable)

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
            A vector of per-trial log-values (exchangeable mode) or a scalar
            (non-exchangeable mode).
        """
        value = pt.cast(pt.as_tensor_variable(value), pytensor.config.floatX)

        if self.exchangeable:
            # Ensure at least 1-D so scalar obs get a batch axis.
            # Preserve any trailing observation dimensions (e.g. (n_obs, obs_dim)).
            if value.ndim == 0:
                value = value[None]
            n_obs = value.shape[0]
            prepared_params = []
            for p in dist_params:
                p = pt.cast(pt.as_tensor_variable(p), pytensor.config.floatX)
                p = pt.reshape(p, (-1,))  # scalar → (1,)
                p = pt.broadcast_to(p, (n_obs,))  # → (n_obs,)
                prepared_params.append(p)
        else:
            prepared_params = [pt.cast(pt.as_tensor_variable(p), pytensor.config.floatX) for p in dist_params]

        return self.backend.logp_op(value, *prepared_params)

    def random(self, *dist_params, rng=None, size=None):
        """Draw random samples (requires ``simulator_fn`` at construction time)."""
        if self.simulator_fn is None:
            raise NotImplementedError("No simulator_fn was provided, so random sampling is unavailable.")
        return self.simulator_fn(rng, *dist_params, size=size)

    def __call__(self, name: str, *args, observed: ArrayLike, **kwargs) -> pm.Distribution:
        """
        Register this distribution as a named variable in the active PyMC model.

        Parameters
        ----------
        name : str
            Name of the PyMC random variable.
        *args :
            Parameters passed positionally (in ``param_names`` order).
        observed : array-like
            Observed data to condition on.
        **kwargs :
            Parameters passed by keyword.  Cannot be mixed with positional args.

        Returns
        -------
        pm.Distribution
        """
        if observed is None:
            raise ValueError("NeuralDistribution requires observed data.")

        params = self._prepare_params(args, kwargs)

        if self.exchangeable:
            signature = ",".join(["()"] * len(self.param_names)) + "->()"
        else:
            ndim = np.asarray(observed).ndim
            dim_names = [chr(ord("m") + i) for i in range(ndim)]
            data_core = "(" + ",".join(dim_names) + ")"
            signature = data_core + "," + ",".join(["()"] * len(self.param_names)) + "->()"

        custom_kwargs = {
            "logp": self.logp,
            "signature": signature,
            "dtype": keras.backend.floatx(),
        }

        if self.simulator_fn is not None:
            custom_kwargs["random"] = self.random

        return pm.CustomDist(name, *params, observed=observed, **custom_kwargs)
